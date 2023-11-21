#TODO
import torch
from torchvision import datasets, transforms
def get_dataset(batch_size, num_workers):
    train_loader = torch.utils.data.DataLoader(
        datasets.GTSRB(
            "./data",
            split="train",
            download=True,
            transform=transforms.Compose([transforms.ToTensor(),]),
        ),
        batch_size=batch_size,
        shuffle=True, 
        num_workers=num_workers,
    )
    test_loader = torch.utils.data.DataLoader(
        datasets.GTSRB(
            "./data",
            split="test",
            download=True,
            transform=transforms.Compose([transforms.ToTensor(),]),
        ),
        batch_size=batch_size,
        shuffle=False, 
        num_workers=num_workers,
    )
    return train_loader,test_loader

