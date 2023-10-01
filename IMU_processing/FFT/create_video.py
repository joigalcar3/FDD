#!/usr/bin/env python3
"""
Function that creates a video given the images of a flight.

It takes as input the images located in the "img_folder" folder and produces the final video in the "out_folder" folder.
It can be used for presentations.
"""

__author__ = "Jose Ignacio de Alvear Cardenas (GitHub: @joigalcar3)"
__copyright__ = "Copyright 2022, Jose Ignacio de Alvear Cardenas"
__credits__ = ["Jose Ignacio de Alvear Cardenas"]
__license__ = "MIT"
__version__ = "1.0.2 (21/12/2022)"
__maintainer__ = "Jose Ignacio de Alvear Cardenas"
__email__ = "jialvear@hotmail.com"
__status__ = "Stable"

# Imports
import os
import cv2
from skimage.io import imread


def create_video(img_folder, fps_rate_in=30, fps_rate_out=30, start_frame=0, out_folder=None, out_file_name=None):
    """
    Function that creates a video in .avi format out of the images of a flight.
    :param img_folder: the path to the location of the images of the flight
    :param fps_rate_in: the frames per second at which the images of the flight were taken
    :param fps_rate_out: the frames per secons at which the output video should be created
    :param start_frame: the first frame used for the generation of the video
    :param out_folder: the folder where output video will be saved
    :param out_file_name: the name of the output video
    :return: the folder where the video will be saved, the name of the output video file and the complete file path to
    the video
    """
    # Obtain the names of the images to be assembled in a video and the size of the images
    img_names = sorted(os.listdir(img_folder))
    image_path = os.path.join(img_folder, img_names[0])
    image = imread(image_path)
    frameSize = image.shape[1::-1]

    # Create the output name and the output folder
    if out_file_name is None:
        out_file_name = img_folder.split("\\")[-1] + f"_s{start_frame}_f{fps_rate_out}"
    if out_folder is None:
        cwd = os.path.dirname(os.getcwd())
        out_folder = f'{cwd}\\video_storage\\' + img_folder.split("\\")[-1]
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    file_address = os.path.join(out_folder, f'{out_file_name}.avi')

    # Create the video writer
    if not os.path.exists(file_address):
        out = cv2.VideoWriter(file_address, cv2.VideoWriter_fourcc(*'DIVX'), fps_rate_out, frameSize)

        # Add each of the images to the video
        for filename in img_names[start_frame::int(fps_rate_in/fps_rate_out)]:
            image_path = os.path.join(img_folder, filename)
            img = cv2.imread(image_path)
            out.write(img)

        # Release the video and save it
        out.release()
    return out_folder, out_file_name, file_address


if __name__ == "__main__":
    images_folder = "E:\\Aliasing\\20221109-214327_1_with\\front"
    output_folder = "E:\\Aliasing"
    output_file_name = "temp"
    create_video(images_folder, fps_rate_in=30, fps_rate_out=30, start_frame=0, out_folder=output_folder,
                 out_file_name=output_file_name)
