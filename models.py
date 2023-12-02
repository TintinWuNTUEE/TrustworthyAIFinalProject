import torch
import timm
from torchsummary import summary
from thop import profile
import torch.nn as nn

def get_model(name):
    if name == 'resnet18':
        model = torch.hub.load('pytorch/vision:v0.9.0', 'resnet18', pretrained=False)
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, 43).cuda()
    elif name == 'resnet50':
        model = torch.hub.load('pytorch/vision:v0.9.0', 'resnet50', pretrained=False)
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, 43).cuda()
    elif name == 'vit-t':
        model = timm.create_model('vit_tiny_patch16_224.augreg_in21k_ft_in1k', pretrained=False)
        num_ftrs = model.head.in_features
        model.head = torch.nn.Linear(num_ftrs, 43).cuda()
    elif name == 'vit-s':
        model = timm.create_model('vit_small_patch16_224.augreg_in21k_ft_in1k', pretrained=False)
        num_ftrs = model.head.in_features
        model.head = torch.nn.Linear(num_ftrs, 43).cuda()
    elif name == 'hybrid-t':
        model = timm.create_model('vit_tiny_r_s16_p8_224.augreg_in21k_ft_in1k', pretrained=False)
        num_ftrs = model.head.in_features
        model.head = torch.nn.Linear(num_ftrs, 43).cuda()
    elif name == 'hybrid-s':
        model = timm.create_model('vit_small_r26_s32_224.augreg_in21k_ft_in1k', pretrained=False)
        num_ftrs = model.head.in_features
        model.head = torch.nn.Linear(num_ftrs, 43).cuda()
    else:
        raise NotImplementedError
    return model

def print_model(model):
    summary(model, (3, 224, 224))
    input = torch.randn(1, 3, 224, 224)
    flops, params = profile(model, inputs=(input, ))
    print(model)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"FLOPs: {flops}, Params: {params}")
    

if __name__ == "__main__":
    model = get_model('resnet18')
    #print_model(model)

    model = get_model('resnet50')
    print_model(model)

    model = get_model('vit-t')
    print_model(model)

    model = get_model('vit-s')
    print_model(model)

    model = get_model('hybrid-t')
    print_model(model)

    model = get_model('hybrid-s')
    print_model(model)


