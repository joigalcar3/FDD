import tensorflow as tf
from datetime import datetime


def convert_to_dataset(generator, BATCH_SIZE, switch_shuffle_buffer=True, **kwargs):
    """
    Converts a generator to a tf.data.Dataset ready to be fed to a model
    :param generator: the generator class
    :param BATCH_SIZE: the size of the batch
    (https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch)
    :param switch_shuffle_buffer: whether the shuffle buffer should be activated
    :param kwargs: the inputs to the generator
    :return: the training and validation dataset objects
    """
    train_generator = generator(generator_type="train", **kwargs)
    data_sample_shape = train_generator.example[0].shape
    label_sample_shape = train_generator.example[1].shape
    output_signature = (tf.TensorSpec(shape=(*data_sample_shape,), dtype=tf.float32),
                        tf.TensorSpec(shape=(*label_sample_shape,), dtype=tf.int8))
    train_ds = tf.data.Dataset.from_generator(train_generator, output_signature=output_signature)

    # Create the validation set
    val_generator = generator(generator_type="val", **kwargs)
    val_ds = tf.data.Dataset.from_generator(val_generator, output_signature=output_signature)

    # Adding the batch dimension
    # https://www.tensorflow.org/guide/data_performance
    # https://www.tensorflow.org/datasets/performances
    # https://www.tensorflow.org/tutorials/load_data/video
    AUTOTUNE = tf.data.AUTOTUNE
    if switch_shuffle_buffer:
        train_ds = train_ds.shuffle(buffer_size=BATCH_SIZE*10, reshuffle_each_iteration=True)
        val_ds = val_ds.shuffle(buffer_size=BATCH_SIZE*10, reshuffle_each_iteration=True)

    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE).batch(BATCH_SIZE)
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE).batch(BATCH_SIZE)

    # Print the shapes of the data
    # train_frames, train_labels = next(iter(train_ds))
    print(f'Shape of training set of frames: {(BATCH_SIZE, *list(data_sample_shape))}')
    print(f'Shape of training labels: {(BATCH_SIZE, *list(label_sample_shape))}')

    # val_frames, val_labels = next(iter(val_ds))
    print(f'Shape of validation set of frames: {(BATCH_SIZE, *list(val_generator.example[0].shape))}')
    print(f'Shape of validation labels: {(BATCH_SIZE, *list(val_generator.example[1].shape))}')

    return train_ds, val_ds, data_sample_shape, [train_generator, val_generator]


def define_callbacks(log_directory, patience, checkpoint_name):
    """
    Function to define all the callbacks for training a model
    :param log_directory: the location of the log directory where the tensorboard files will be stored
    :param patience: how many epochs to wait without improving the loss function before early stopping
    :param checkpoint_name: the name of the checkpoint
    :return:
    """
    datestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    logdir = f'{log_directory}/fit/{checkpoint_name}_{datestamp}'
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir, update_freq="batch")

    early_stop_callback = tf.keras.callbacks.EarlyStopping(patience=patience, monitor='val_loss', verbose=1)

    checkpoint_logdir = f'checkpoints/{checkpoint_name + datestamp}'
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_logdir, verbose=1,
                                                                   monitor="val_loss", save_best_only=True,
                                                                   save_weights_only=True)
    csv_logger_callback = tf.keras.callbacks.CSVLogger("csv_logs/training.csv")

    callbacks = [tensorboard_callback, early_stop_callback, model_checkpoint_callback, csv_logger_callback]

    # Tensorboard batch logging
    log_dir = f'{log_directory}/batch_level/{checkpoint_name}_{datestamp}/train'
    train_writer = tf.summary.create_file_writer(log_dir)
    batch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('batch_accuracy')

    return callbacks, train_writer, batch_accuracy


# https://www.tensorflow.org/tensorboard/scalars_and_keras
class BatchLogging(tf.keras.Model):
    """
    Class used for saving batch information for posterior analysis.
    If the saved model in a checkpoint uses this class during training, this class must also be used for evaluation
    when the weights are loaded.
    """
    def __init__(self, model, train_writer, batch_accuracy):
        super().__init__()
        self.model = model
        self.train_writer = train_writer
        self.batch_accuracy = batch_accuracy

    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            y_pred = self.model(x, training=True)
            loss = self.compiled_loss(y, y_pred)
        self.optimizer.minimize(loss, self.trainable_variables, tape=tape)
        batch_acc = self.batch_accuracy(y, y_pred)
        internal_metrics = {m.name: m.result() for m in self.metrics}
        with self.train_writer.as_default(step=self._train_counter):
            tf.summary.scalar('batch_loss', internal_metrics["loss"])
            tf.summary.scalar('batch_accuracy', batch_acc)
        return self.compute_metrics(x, y, y_pred, None)

    def call(self, x):
        x = self.model(x)
        return x


# https://stackoverflow.com/questions/2130016/splitting-a-list-into-n-parts-of-approximately-equal-length?page=1&tab=scoredesc#tab-top
def split_indices(a, n):
    k, m = divmod(len(a), n)
    return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))



