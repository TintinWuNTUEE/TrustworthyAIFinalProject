import torch
import torch.nn as nn
import torchvision.models as models
from attention_rollout import VITAttentionGradRollout
import timm
if __name__ == "__main__":
    import timm
    model = timm.create_model('vit_small_patch16_224.augreg_in1k', pretrained=True)
#     model = torch.hub.load('facebookresearch/deit:main', 
# 'deit_tiny_patch16_224', pretrained=True)
    grad_rollout_ = VITAttentionGradRollout(model, discard_ratio=0.9)
    input_tensor = torch.rand(1, 3, 224, 224)
    mask = grad_rollout_(input_tensor,category_index=243)

