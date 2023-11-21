from transformers import AutoFeatureExtractor, ResNetForImageClassification
import torch
import timm

def get_model(name):
    if name == 'resnet18':
        feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/resnet-18")
        model = ResNetForImageClassification.from_pretrained("microsoft/resnet-18")
    elif name == 'resnet50':
        feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/resnet-50")
        model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")
    elif name == 'vit-s':
        model = timm.create_model('vit_small_patch16_224.augreg_in1k', pretrained=True)
    elif name == 'vit-t':
        model = timm.create_model('tiny_vit_11m_224.in1k', pretrained=True)
    else:
        raise NotImplementedError
    return model

    