import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
def get_optimizer(name, model, lr, weight_decay):
    if name == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        # optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    else:
        raise NotImplementedError(f"Optimizer '{name}' is not implemented.")
    return optimizer, scheduler