from transformers import AutoFeatureExtractor, ResNetForImageClassification
import torch
import timm


def get_model(name):
    if name == 'resnet18':
        model = torch.hub.load('pytorch/vision:v0.9.0', 'resnet18', pretrained=True)
    elif name == 'resnet50':
        model = torch.hub.load('pytorch/vision:v0.9.0', 'resnet50', pretrained=True)
    elif name == 'vit-s':
        model = timm.create_model('vit_small_patch16_224.augreg_in1k', pretrained=True)
    elif name == 'vit-t':
        model = timm.create_model('tiny_vit_11m_224.in1k', pretrained=True)
    else:
        raise NotImplementedError
    return model

if __name__ == "__main__":
    model = get_model('resnet18')
    print(model)
    model = get_model('resnet50')
    print(model)
    model = get_model('vit-s')
    print(model)
    model = get_model('vit-t')
    print(model)