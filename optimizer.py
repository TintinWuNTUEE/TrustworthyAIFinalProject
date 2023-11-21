import torch
import torchvision
def get_optimizer(name,lr,weight_decay):
    if name =='Adam':
        optim = torch.optim.Adam(lr=lr,weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optim, T_max=10)
    else: 
        raise NotImplementedError
    return optim,scheduler