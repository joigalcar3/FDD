import os
import cv2
from skimage.io import imread


def create_video(img_folder, fps_rate_in=30, fps_rate_out=30, start_frame=0, out_folder=None, out_file_name=None):
    img_names = sorted(os.listdir(img_folder))
    image_path = os.path.join(img_folder, img_names[0])
    image = imread(image_path)
    frameSize = image.shape[1::-1]
    if out_file_name is None:
        out_file_name = img_folder.split("\\")[-1] + f"_s{start_frame}_f{fps_rate_out}"
    if out_folder is None:
        cwd = os.path.dirname(os.getcwd())
        out_folder = f'{cwd}\\video_storage\\' + img_folder.split("\\")[-1]
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    file_address = os.path.join(out_folder, f'{out_file_name}.avi')
    if not os.path.exists(file_address):
        out = cv2.VideoWriter(file_address, cv2.VideoWriter_fourcc(*'DIVX'), fps_rate_out, frameSize)

        for filename in img_names[start_frame::int(fps_rate_in/fps_rate_out)]:
            image_path = os.path.join(img_folder, filename)
            img = cv2.imread(image_path)
            out.write(img)

        out.release()
    return out_folder, out_file_name, file_address


if __name__ == "__main__":
    img_folder = "E:\\Aliasing\\20221109-214327_1\\front"
    out_folder = "E:\\Aliasing"
    out_file_name = "with_anti_aliasing"
    create_video(img_folder, fps_rate_in=30, fps_rate_out=30, start_frame=0, out_folder=out_folder, out_file_name=out_file_name)
