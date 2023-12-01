import cv2
import numpy as np
from skimage.feature import hog

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
                                   visualize=False)
    return normalised_blocks, hog_image
def mask_hog(img):
    _,hog_mask = get_hog(img)
    hog_mask_rgb = cv2.cvtColor(hog_mask, cv2.COLOR_GRAY2RGB)
    return img+hog_mask_rgb