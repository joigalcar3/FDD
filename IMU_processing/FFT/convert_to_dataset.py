import tensorflow as tf


def convert_to_dataset(generator, BATCH_SIZE, **kwargs):
    """
    Converts a generator to a tf.data.Dataset ready to be fed to a model
    :param generator: the generator class
    :param BATCH_SIZE: the size of the batch
    :param kwargs: the inputs to the generator
    :return: the training and validation dataset objects
    """
    train_generator = generator(generator_type="train", **kwargs)
    data_sample_shape = train_generator.example[0].shape
    output_signature = (tf.TensorSpec(shape=(*data_sample_shape,), dtype=tf.float32),
                        tf.TensorSpec(shape=(data_sample_shape[0], 1), dtype=tf.int8))
    train_ds = tf.data.Dataset.from_generator(train_generator, output_signature=output_signature)

    # Create the validation set
    val_generator = generator(generator_type="val", **kwargs)
    val_ds = tf.data.Dataset.from_generator(val_generator, output_signature=output_signature)

    # Adding the batch dimension
    train_ds = train_ds.batch(BATCH_SIZE)
    val_ds = val_ds.batch(BATCH_SIZE)

    # Print the shapes of the data
    # train_frames, train_labels = next(iter(train_ds))
    print(f'Shape of training set of frames: {(BATCH_SIZE, *list(train_generator.example[0].shape))}')
    print(f'Shape of training labels: {(BATCH_SIZE, *list(train_generator.example[1].shape))}')

    # val_frames, val_labels = next(iter(val_ds))
    print(f'Shape of validation set of frames: {(BATCH_SIZE, *list(val_generator.example[0].shape))}')
    print(f'Shape of validation labels: {(BATCH_SIZE, *list(val_generator.example[1].shape))}')

    return train_ds, val_ds, data_sample_shape, [train_generator, val_generator]
