import numpy as np
import cv2


def random_crop(image, gt_image, out_width, out_height, p=0.5):
    """Randomly cropping augmentation. The cropping executes around the ball blob location."""
    if np.random.uniform() > p:
        return {'image': image, 'gt': gt_image}
    else:
        # find center of ball's blob:
        ball_blob = np.array(np.nonzero(gt_image)).T
        c_y, c_x = np.mean(ball_blob[:, 0]), np.mean(ball_blob[:, 1])
        r, l, h, b = [None] * 4
        img_h, img_w = image.shape[:-1]

        cropped_img_w = image.copy()
        cropped_gt_image_w = gt_image.copy()
        if img_w != out_width:
            flag_width = False
            while not flag_width:  # loop until get valid corners
                l = np.random.randint(low=0, high=c_x)
                r = l + out_width
                if r < img_w:
                    flag_width = True
            cropped_img_w = image[:, l:r, :]
            cropped_gt_image_w = gt_image[:, l:r]

        cropped_img = cropped_img_w
        cropped_gt = cropped_gt_image_w
        if img_h != out_height:
            flag_height = False
            while not flag_height:  # loop until get valid corners
                h = np.random.randint(low=0, high=c_y)
                b = h + out_height
                if b < img_h:
                    flag_height = True
            cropped_img = cropped_img_w[h:b, :, :]
            cropped_gt = cropped_gt_image_w[h:b, :]

        return {'image': cropped_img, 'gt': cropped_gt}


def flip_lr(image, gt_image, p=0.5):
    if np.random.uniform() > p:
        return {'image': image, 'gt': gt_image}
    else:
        flipped_img = cv2.flip(image, 1)
        flipped_gt = cv2.flip(gt_image, 1)
        return {'image': flipped_img, 'gt': flipped_gt}