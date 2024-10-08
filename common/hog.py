import cv2
import numpy as np
from skimage.feature import hog
import torch
import sys 
from .utils import pil2cv2_grayscale
# from dataset import get_dataset

import time

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

if __name__ == "__main__":
    train_loader,_ = get_dataset(1,0)
    images, labels = next(iter(train_loader))
    # image = images[0].permute(1,2,0).numpy()
    
    print(time.ctime(time.time())) 
    image = mask_hog(images[0])
    print(time.ctime(time.time())) 
    cv2.imwrite('hog_mask.jpg',image)