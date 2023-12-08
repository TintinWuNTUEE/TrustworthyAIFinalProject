import cv2
import numpy as np
def pil2cv2_grayscale(img):
    img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2GRAY)
    return img