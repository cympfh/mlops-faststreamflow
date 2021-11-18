from typing import Optional

import torchvision
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST


class Dataset:

    train_dataloader: DataLoader
    val_dataloader: Optional[DataLoader]

    def __init__(self, train_dataloader, val_dataloader):
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

    @classmethod
    def load(cls):
        transform = torchvision.transforms.ToTensor()
        train_dataset = MNIST(root="data", download=True, transform=transform)
        val_dataset = MNIST(root="data", train=False, download=True, transform=transform)
        train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        return cls(train_dataloader, val_dataloader)
