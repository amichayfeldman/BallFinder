import os
import numpy as np
import random
import pandas as pd
import glob


def split_data_into_sets(main_images_folder, main_gt_folder):
    all_data = pd.DataFrame(columns=['GT', 'image_path'])
    all_data['GT'] = glob.glob(main_gt_folder + '/**/*.jpg', recursive=True)

