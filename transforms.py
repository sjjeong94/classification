import torch
import random
import numpy as np


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image):
        for transform in self.transforms:
            image = transform(image)
        return image


class Pad:
    def __init__(self, padding):
        self.padding = padding

    def __call__(self, image):
        p = self.padding
        pad_width = ((p, p), (p, p), (0, 0))
        image = np.pad(image, pad_width)
        return image


class RandomCrop:
    def __init__(self, size):
        self.size = size

    def __call__(self, image):
        h, w, c = image.shape
        xs = random.randint(0, w - self.size)
        ys = random.randint(0, h - self.size)
        xe = xs + self.size
        ye = ys + self.size
        image = image[ys:ye, xs:xe]
        return image


class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image):
        if random.random() < self.p:
            image = np.fliplr(image)
        return image


class ToTensor:
    def __call__(self, image):
        image = image.transpose(2, 0, 1)
        image = image.astype(np.float32) / 255
        image = torch.from_numpy(image)
        return image


class Normalize:
    def __init__(self, mean, std):
        self.mean = torch.FloatTensor(mean).reshape(-1, 1, 1)
        self.std = torch.FloatTensor(std).reshape(-1, 1, 1)

    def __call__(self, image):
        image = (image - self.mean) / self.std
        return image
