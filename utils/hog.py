import cv2
import numpy as np
from skimage.feature import hog

import torch
def pil2cv2_grayscale(img):
    img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2GRAY)
    return img
def get_hog(img):
    img_gray = pil2cv2_grayscale(img)
    normalised_blocks, hog_image = hog(img_gray,
                                   orientations=9,
                                   pixels_per_cell=(8, 8),
                                   cells_per_block=(2, 2),
                                   block_norm='L2-Hys',
                                   visualize=True)
    return normalised_blocks, hog_image
def mask_hog(img):
    # print(img.size())
    if torch.cuda.is_available():
        img=np.transpose(img.cpu().detach().numpy(),(1,2,0))
    else:
        img=np.transpose(img.detach().numpy(),(1,2,0))
    _,hog_mask = get_hog(img)
    # hog_mask_rgb = cv2.cvtColor(hog_mask, cv2.COLOR_GRAY2RGB)
    return hog_mask
    # return hog_mask_rgb#,hog_mask,img+hog_mask_rgb