import cv2
import numpy as np
import torch
import torchvision.models as models
from skimage.feature import hog


# Compute HOG features from the obtained features
def compute_hog_features(features):
    hog_features = []
    for feature in features:
        hog_feature, _ = hog(feature, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
        hog_features.append(hog_feature)
    return hog_features
def HOG(model,image):
    # Get the features from the last convolutional layer
    with torch.no_grad():
        features = model(image)

    # Convert the features to numpy array
    features = features.numpy()

    # Compute the HOG features
    hog_features = compute_hog_features(features)
    return hog_features
