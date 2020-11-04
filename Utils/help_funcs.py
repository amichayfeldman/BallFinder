import csv
import os
import glob
import cv2
from pathlib import Path
import math
from itertools import zip_longest


def write_to_csv(csv_path, lists):
    export_data = zip_longest(*lists, encoding="ISO-8859-1", newline='')
    with open(csv_path, 'w') as file:
        wr = csv.writer(file)
        wr.writerow(("Epoch", "Train_loss", "Val_loss", "lr", "wd"))
        wr.writerow(export_data)
    file.close()


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
    col_start_idx, row_start_idx = -patch_w, 0
    col_end_idx, row_end_idx = 0, patch_h

    switch_row = False
    break_flag = False
    while not break_flag:
        if x_shape[2] <= patch_h or x_shape[3] <= patch_w:
            yield 0, x_shape[2], 0, x_shape[3]
            break_flag = True

        if switch_row:
            if row_end_idx + patch_h < x_shape[2]:
                row_start_idx += patch_h
                row_end_idx += patch_h
            else:
                row_start_idx = -patch_h
                row_end_idx = x_shape[2]
            col_start_idx, col_end_idx = -patch_w, 0
            switch_row = False

        if col_end_idx + patch_w < x_shape[3]:
            col_start_idx += patch_w
            col_end_idx += patch_w

        elif col_end_idx + patch_w >= x_shape[3]:
            col_start_idx = -patch_w
            col_end_idx = x_shape[3]
            switch_row = True

        if col_end_idx == x_shape[3] and row_end_idx == x_shape[2]:
            break_flag = True

        yield row_start_idx, row_end_idx, col_start_idx, col_end_idx


def plot_dataloader(dataloader, output_folder):
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)

    for idx, data in enumerate(dataloader):
        inputs, labels, indices, img_paths = data['image'], data['gt'], data['idx'], data['img_path']
        for s in range(inputs.shape[0]):
            img = inputs[s, ...].detach().cpu().permute(1, 2, 0)
            gt_map = labels[s, ...].detach().cpu().numpy()
            index = indices[s]
            img_path = img_paths[s]
            img_name = os.path.basename(img_path)
            img_num = img_name.split('.')[0].split('_')[1]
            img = cv2.cvtColor(255 * img.numpy().astype('float32'), cv2.COLOR_RGB2BGR)
            gt_map = 255 * gt_map.astype('uint8')
            gt_map[0, :] = 255
            gt_map[-1, :] = 255
            gt_map[:, 0] = 255
            gt_map[:, -1] = 255
            cv2.imwrite(os.path.join(output_folder, 'img_{}.jpg'.format(img_num)), cv2.resize(src=img, dsize=(354, 200)))
            # cv2.imwrite(os.path.join(output_folder, 'segmap_{}.jpg'.format(img_num)),
            #             cv2.resize(src=gt_map, dsize=(177, 100)))


# convert_png_folder_to_jpg('/home/amichay/DL/BallDetector/Dataset/GroundTruth/')