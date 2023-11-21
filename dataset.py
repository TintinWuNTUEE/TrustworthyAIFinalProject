#TODO
import torch
from torchvision import datasets, transforms
def get_dataset(batch_size, num_workers):
    dataset = None
    test_loader = torch.utils.data.DataLoader(
        datasets.GTSRB(
            "./data",
            train=False,
            download=True,
            transform=transforms.Compose([transforms.ToTensor(),]),
        ),
        batch_size=batch_size,
        shuffle=True, 
        num_workers=num_workers,
    )
    return dataset 