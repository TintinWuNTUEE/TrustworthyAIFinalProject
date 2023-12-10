
import cv2
import numpy as np
from dataset import get_dataset
import time
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
# Download the checkpoint from https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
# and put it in the same directory as this file.
CHKPT = './SAM/sam_vit_h_4b8939.pth'
def create_segmentor():
    '''
    Create a segmentor that segments the largest object in the image.
    This is the model, so that we don't have to load it every time we want to segment an image
    '''
    sam = sam_model_registry["vit_h"](checkpoint=CHKPT)
    mask_generator = SamAutomaticMaskGenerator(sam)
    return mask_generator
def get_segment(image,mask_generator):
    '''
    Get the largest object in the image
    Mask it out on the orginal image
    '''
    image = np.uint8(255*image)
    masks = mask_generator.generate(image)
    max_area_dict = max(masks, key=lambda x: x['area'])
    max_area_mask = max_area_dict['segmentation']
    image[max_area_mask] = 0
    return image
if __name__ == "__main__":
    train_loader,_ = get_dataset(1,0)
    images, labels = next(iter(train_loader))
    image = images[0].permute(1,2,0).numpy()
    mask_generator = create_segmentor()
    print(time.ctime(time.time())) 
    image = get_segment(image,mask_generator)
    print(time.ctime(time.time())) 
    cv2.imwrite('mask.jpg',image)