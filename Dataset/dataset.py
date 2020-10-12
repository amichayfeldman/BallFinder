import cv2
import os
import torch
import pandas as pd
import glob
import re
import numpy as np
import imgaug.augmenters as iaa
from Dataset.Augmentations import random_crop, flip_lr
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt

sometimes = lambda aug: iaa.Sometimes(0.5, aug)


class BallDataset(torch.utils.data.Dataset):
    def __init__(self, images_list, gt_list, img_h, img_w, mode='train'):
        self.data = pd.DataFrame(columns=['img_path', 'gt_path'])
        self.data['img_path'] = images_list
        self.gt_imgs_list = gt_list
        self.data = self.data.apply(self.assign_gt_to_img, axis=1)
        self.data.dropna(axis=0, inplace=True)
        self.data.reset_index(drop=True, inplace=True)

        self.img_w, self.img_h = img_w, img_h
        self.mode = mode

    def assign_gt_to_img(self, row):
        frame_number = "".join(re.findall("\d+", os.path.basename(row['img_path'])))
        folder_num = row['img_path'].split('/')[-2][-1]
        adjusted_gt = [x for x in self.gt_imgs_list if frame_number in x and x.split('/')[-3][-1] == folder_num]
        if len(adjusted_gt) > 0:
            row['gt_path'] = adjusted_gt[0]
        else:
            row['gt_path'] = None
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
        gt_mask[gt_mask > 0] = 1

        if self.mode == 'test':
            # tensored_image = torch.from_numpy(image)
            # tensored_gt = torch.from_numpy(gt_mask)
            sample = {'image': np.moveaxis(image, -1, 0) / 255, 'gt': gt_mask, 'idx': idx}
            return sample
        else:
            seq = iaa.Sequential([iaa.OneOf([iaa.GaussianBlur((0, 3.0)),
                                             iaa.AverageBlur(k=(2, 7)),
                                             # iaa.MedianBlur(k=(3, 11))
                                             ]),
                                  sometimes(iaa.OneOf([iaa.Add((-10, 10), per_channel=0.5),
                                            iaa.Multiply((0.85, 1.15), per_channel=0.5)]))
                                  ])

            augmented_img = seq(image=image)
            flipped_or_not = flip_lr(image=augmented_img, gt_image=gt_mask)
            after_crop = random_crop(image=flipped_or_not['image'], gt_image=flipped_or_not['gt'],
                                     out_width=self.img_w, out_height=self.img_h)

            # tensored_image = torch.from_numpy(after_crop['image'])
            # tensored_gt = torch.from_numpy(after_crop['gt'])
            # sample = {'image': tensored_image.permute(2, 0, 1) // 255, 'gt': tensored_gt, 'idx': idx}
            sample = {'image': np.moveaxis(after_crop['image'], -1, 0) / 255, 'gt': after_crop['gt'], 'idx': idx}
            return sample


def get_dataloaders(dataset_dict, gt_dict, batch_size, num_workers, shuffle=True):
    """
    Get train, val and test dataloaders of Ballfinder.
    :param dataset_dict: Dict. Paths for dataset input images with structure of
    {'train':<list>, 'val':<list>, 'test':<list>}
    :param gt_dict: Dict. Paths for dataset input images with structure of
    {'train':<list>, 'val':<list>, 'test':<list>}
    :param batch_size: int.
    :param num_workers: int.
    :param shuffle: Boolean.
    :return: Three Dataloaders for training.
    """
    print("# - # - # - # - # - # - # - # - # - # - # - # - # - #")
    if 'train' in dataset_dict:
        print("Building train set", end="")
        train_set = BallDataset(dataset_dict['train'], gt_dict['train'], img_h=720, img_w=1280, mode='train')
        print("...")
        train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=shuffle,
                                                       num_workers=num_workers)
        print("Finished the train dataloader building")
    else:
        train_dataloader = None

    if 'val' in dataset_dict:
        print("Building val set", end="")
        val_set = BallDataset(dataset_dict['val'], gt_dict['val'], img_h=720, img_w=1280, mode='val')
        print("...")
        val_dataloader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=shuffle,
                                                       num_workers=num_workers)
        print("Finished the val dataloader building")
    else:
        val_dataloader = None

    if 'test' in dataset_dict:
        print("Building val set", end="")
        test_set = BallDataset(dataset_dict['test'], gt_dict['test'], img_h=720, img_w=1280, mode='test')
        print("...")
        test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False,
                                                       num_workers=num_workers)
        print("Finished the test dataloader building")
    else:
        test_dataloader = None

    print("# - # - # - # - # - # - # - # - # - # - # - # - # - #")
    print("")
    return train_dataloader, val_dataloader, test_dataloader

