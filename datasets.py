import torch
import numpy as np
from torchvision import datasets

classes = ['airplane', 'automobile', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


class CIFAR10(torch.utils.data.Dataset):
    def __init__(self, root, train=True, transform=None, download=False):
        super().__init__()
        self.dataset = datasets.CIFAR10(root, train, download=download)
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        image = np.asarray(image)
        if self.transform:
            image = self.transform(image)
        return image, label


class CIFAR100(torch.utils.data.Dataset):
    def __init__(self, root, train=True, transform=None, download=False):
        super().__init__()
        self.dataset = datasets.CIFAR100(root, train, download=download)
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        image = np.asarray(image)
        if self.transform:
            image = self.transform(image)
        return image, label
