import numpy as np
import cv2 
import timm

from dataset import get_dataset
from utils.attention_rollout import VITAttentionGradRollout

def show_mask_on_image(img, mask):
    img = np.float32(img) / 255
    
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    print(heatmap.shape)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)
def get_attention_map(image, model, category_index):
    '''
    image: torch.Tensor, shape: (1, 3, H, W)
    model: torch.nn.Module
    category_index: int, the index of the category to visualize
    '''
    grad_rollout_ = VITAttentionGradRollout(model, discard_ratio=0.9)
    mask = grad_rollout_(image,category_index=category_index)
    image = image.squeeze(0).permute(1,2,0).numpy()
    mask = cv2.resize(mask, (image.shape[0], image.shape[1]))
    mask = show_mask_on_image(image, mask)
    return mask
if __name__ == "__main__":
    import timm
    model = timm.create_model('vit_small_patch16_224.augreg_in1k', pretrained=True)
    model.eval()
    train_loader,_ = get_dataset(1,0)
    images, labels = next(iter(train_loader))
    image = images[0].unsqueeze(0)
    mask = get_attention_map(image, model, 0)
    print(mask.shape)
    mask = show_mask_on_image(image, mask)
    cv2.imshow("Input Image", image)
    cv2.imshow('mask', mask)
    cv2.waitKey(-1)
