import csv
import os
import glob
import cv2
from pathlib import Path
import math


def write_to_csv(csv_path, list_to_write):
    with open(csv_path, 'w') as file:
        wr = csv.writer(file, quoting=csv.QUOTE_ALL)
        wr.writerow(list_to_write)


def convert_png_folder_to_jpg(input_folder_path):
    png_list = list(Path(input_folder_path).rglob("*.png"))

    for file in png_list:
        img = cv2.imread(os.path.join(str(file.parent), file.name))
        name = file.name.split('.')[0]
        cv2.imwrite('{}.jpg'.format(os.path.join(str(file.parent), name)), img)


def divide_input_to_patches(x_shape, config):
    """
    Yield indices of cropped tensor for feeding it to NN in slices(patches)
    :param x_shape: List of tensor shape.
    :param config: Configparser object
    :return: Yield each iteration the start & end indices of rows and columns.
    """
    patch_w, patch_h = config.getint('Params', 'patch_w'), config.getint('Params', 'patch_h')
    col_start_idx, row_start_idx = -int(patch_w / 2), 0
    col_end_idx, row_end_idx = int(patch_w / 2), patch_h

    break_flag = False
    while not break_flag:
        if x_shape[2] <= patch_h or x_shape[3] <= patch_w:
            yield row_start_idx, row_end_idx, col_start_idx, col_end_idx
            break_flag = True

        # print(row_start_idx, "",  row_end_idx, "", col_start_idx, "",  col_end_idx)
        if col_end_idx == x_shape[3]:  # end single row
            col_start_idx, col_end_idx = 0, patch_w
            if row_end_idx == x_shape[2]:
                break_flag = True
            elif row_end_idx + int(patch_h / 2) > x_shape[2]:
                # stride_h = row_end_idx + int(patch_h / 2) - x_shape[2]
                # row_start_idx, row_end_idx = row_start_idx + stride_h, row_end_idx + stride_h
                row_start_idx, row_end_idx = -patch_h, x_shape[2]
            else:
                row_start_idx, row_end_idx = row_start_idx + int(patch_h / 2), row_end_idx + int(patch_h / 2)

        elif col_end_idx + int(patch_w / 2) > x_shape[3]:  # stride will be smaller than patch_w / 2
            # stride_w = col_end_idx + int(patch_w / 2) - x_shape[3]
            # col_start_idx, col_end_idx = col_start_idx - stride_w, col_end_idx - stride_w
            col_start_idx, col_end_idx = -patch_w, x_shape[3]
        else:
            col_start_idx, col_end_idx = col_start_idx + int(patch_w / 2), col_end_idx + int(patch_w / 2)

        yield row_start_idx, row_end_idx, col_start_idx, col_end_idx


# convert_png_folder_to_jpg('/home/amichay/DL/BallDetector/Dataset/GroundTruth/')