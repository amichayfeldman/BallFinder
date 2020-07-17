import cv2
import os
import torch
import pandas as pd
import glob
import re
import numpy as np
import imgaug.augmenters as iaa
from Augmentations import random_crop


class BallDataset(torch.utils.data.Dataset):
    def __init__(self, images_folder, gt_folder_path, img_h, img_w, mode='train'):
        self.data = pd.DataFrame(columns=['img_path', 'gt_path'])
        self.data['img_path'] = glob.glob(os.path.join(images_folder, '*.png'))
        self.gt_imgs_list = glob.glob(os.path.join(gt_folder_path, '*.png'))
        self.data = self.data.apply(self.assign_gt_to_img, axis=1)
        self.data.dropna(axis=0, inplace=True)

        self.img_w, self.img_h = img_w, img_h
        self.mode = mode

    def assign_gt_to_img(self, row):
        frame_number = "".join(re.findall("\d+", os.path.basename(row['img_path'])))
        adjusted_gt = next((x for x in self.gt_imgs_list if frame_number in x))
        row['gt_path'] = adjusted_gt
        return row

    def __len__(self):
        return len(self.data)

    def change_img_size(self):
        random_w_power = torch.randint(8, np.log2(1280))
        random_h_power = torch.randint(8, np.log2(720))
        self.img_w, self.img_h = 2**random_w_power, 2**random_h_power

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image = cv2.imread(row['img_path'])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        gt_mask = cv2.imread(row['gt_path'], cv2.IMREAD_GRAYSCALE)

        if self.mode == 'test':
            tensored_image = torch.from_numpy(image)
            tensored_gt = torch.from_numpy(gt_mask)
            sample = {'image': tensored_image.permute(2, 0, 1), 'gt':tensored_gt.permute(2, 0, 1), 'idx':[idx]}
            return sample
        else:
            seq = iaa.Sequential([iaa.Fliplr(0.5),
                                  iaa.OneOf([iaa.GaussianBlur((0, 5.0)), iaa.AverageBlur(k=(2, 7)),
                                             iaa.MedianBlur(k=(3, 11))]),
                                  iaa.OneOf([iaa.Add((-10, 10), per_channel=0.5),
                                             iaa.Multiply((0.7, 1.3), per_channel=0.5)])])

            augmented_img = seq(images=image)
            after_crop = random_crop(image=augmented_img, gt_image=gt_mask, out_width=self.img_w, out_height=self.img_h)

            tensored_image = torch.from_numpy(after_crop['image'])
            tensored_gt = torch.from_numpy(after_crop['gt'])
            sample = {'image': tensored_image.permute(2, 0, 1), 'gt': tensored_gt.permute(2, 0 , 1), 'idx': [idx]}
            return sample


def get_dataloaders(dataset_folders, gt_folders, batch_size, num_workers, shuffle=True):
    """
    Get train, val and test dataloaders of Ballfinder.
    :param dataset_folders: Dict. Paths for dataset input images with structure of
    {'train':<str>, 'val':<str>, 'test':<str>}
    :param gt_folders: Dict. Paths for dataset input images with structure of
    {'train':<str>, 'val':<str>, 'test':<str>}
    :param batch_size: int.
    :param num_workers: int.
    :param shuffle: Boolean.
    :return: Three Dataloaders for training.
    """
    train_set = BallDataset(dataset_folders['train'], gt_folders['train'], img_h=720, img_w=1280, mode='train')
    val_set = BallDataset(dataset_folders['val'], gt_folders['val'], img_h=720, img_w=1280, mode='val')
    test_set = BallDataset(dataset_folders['test'], gt_folders['test'], img_h=720, img_w=1280, mode='test')
    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=shuffle,
                                                   num_workers=num_workers)
    val_dataloader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=shuffle,
                                                   num_workers=num_workers)
    test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False,
                                                   num_workers=num_workers)
    return train_dataloader, val_dataloader, test_dataloader

