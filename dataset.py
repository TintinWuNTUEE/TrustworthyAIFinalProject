#TODO
import torch
from torchvision import datasets, transforms
def get_dataset(batch_size, num_workers):
<<<<<<< HEAD
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
=======
    dataset = None
    test_loader = torch.utils.data.DataLoader(
        datasets.GTSRB(
            "./data",
            train=False,
>>>>>>> main
            download=True,
            transform=transforms.Compose([transforms.ToTensor(),]),
        ),
        batch_size=batch_size,
<<<<<<< HEAD
        shuffle=False, 
        num_workers=num_workers,
    )
    return train_loader,test_loader

=======
        shuffle=True, 
        num_workers=num_workers,
    )
    return dataset 
>>>>>>> main
