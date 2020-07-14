import numpy as np
import cv2


def random_crop(image, gt_image, out_width, out_height, zoom=False, p=0.5):
    if np.random.uniform() > p:
        return {'image': image, 'gt': gt_image}
    else:
        # find center of ball's blob:
        ball_blob = gt_image[gt_image != 0]
        c_y, c_x = np.mean(ball_blob[:, 0]), np.mean(ball_blob[:, 1])
        r, l, h, b = [None] * 4
        img_h, img_w = image.shape[:-1]
        # c_y, c_x = center_pt
        flag_width = False
        while not flag_width:  # loop until get valid corners
            r = np.random.randint(low=0, high=c_x)
            l = out_width - r
            if c_x + l < img_w :
                flag_width = True
        flag_height = False
        while not flag_height:  # loop until get valid corners
            h = np.random.randint(low=0, high=c_y)
            b = out_height - h
            if c_y + b < img_h:
                flag_height = True

        cropped_img = image[c_y-h:c_y+b, c_x-l:c_x+r, :]
        cropped_gt_image = gt_image[c_y-h:c_y+b, c_x-l:c_x+r]

        if zoom:
            cropped_img = cv2.resize(src=cropped_img, dsize=image.shape)
            cropped_gt_image = cv2.resize(src=cropped_gt_image, dsize=image.shape)

        return {'image': cropped_img, 'gt': cropped_gt_image}

