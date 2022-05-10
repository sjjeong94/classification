import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets
import albumentations
import albumentations.pytorch


def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


class CIFAR10(datasets.CIFAR10):
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        if self.transform is not None:
            img = self.transform(image=img)['image']

        return img, target


class CIFAR100(datasets.CIFAR100):
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        if self.transform is not None:
            img = self.transform(image=img)['image']

        return img, target


def get_transform(random_crop=False, random_flip=False, cutout=False):
    tr = []
    if random_crop:
        tr.append(albumentations.PadIfNeeded(36, 36))
        tr.append(albumentations.RandomCrop(32, 32))
    if random_flip:
        tr.append(albumentations.HorizontalFlip())
    if cutout:
        tr.append(albumentations.CoarseDropout(max_holes=1, p=1))
    m, s = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
    tr.append(albumentations.Normalize(m, s))
    tr.append(albumentations.pytorch.ToTensorV2())
    return albumentations.Compose(tr)


def ConvBNReLU(in_ch, out_ch, k_size, stride=1, padding=0):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, k_size, stride, padding, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU()
    )


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k_size, stride=1, padding=0):
        super().__init__()
        self.layers = nn.Sequential(
            ConvBNReLU(in_ch, out_ch, k_size, stride, padding),
            ConvBNReLU(in_ch, out_ch, k_size, stride, padding),
        )

    def forward(self, x):
        return self.layers(x) + x


def Classifier(num_classes):
    return nn.Sequential(
        ConvBNReLU(3, 64, 3, 1, 1),
        ConvBNReLU(64, 128, 3, 1, 1),
        nn.MaxPool2d(2, 2),
        ResBlock(128, 128, 3, 1, 1),
        ConvBNReLU(128, 256, 3, 1, 1),
        nn.MaxPool2d(2, 2),
        ConvBNReLU(256, 512, 3, 1, 1),
        nn.MaxPool2d(2, 2),
        ResBlock(512, 512, 3, 1, 1),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(512, num_classes)
    )


class Trainer:
    def __init__(
        self,
        root='./data',
        dataset='CIFAR10',
        seed=1234,
        epochs=24,
        batch_size=128,
        learning_rate=0.1,
        momentum=0.9,
        weight_decay=5e-4,
        random_crop=True,
        random_flip=True,
        cutout=True,
        amp_enabled=True
    ):
        set_seed(seed)
        self.epochs = epochs

        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        if self.device == 'cpu':
            amp_enabled = False
        self.amp_enabled = amp_enabled

        if dataset == 'CIFAR10':
            num_classes = 10
            load_dataset = CIFAR10
        else:
            num_classes = 100
            load_dataset = CIFAR100

        T_train = get_transform(random_crop, random_flip, cutout)
        T_test = get_transform()

        train_dataset = load_dataset(
            root=root, train=True, transform=T_train, download=True)
        test_dataset = load_dataset(
            root=root, train=False, transform=T_test, download=False)

        self.train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=1,
            pin_memory=True)
        self.test_loader = torch.utils.data.DataLoader(
            dataset=test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=1,
            pin_memory=True)

        self.net = Classifier(num_classes).to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(
            self.net.parameters(),
            lr=learning_rate,
            momentum=momentum, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=learning_rate,
            epochs=epochs,
            steps_per_epoch=len(self.train_loader),
            anneal_strategy='linear')

    def train(self):
        net = self.net
        epochs = self.epochs
        device = self.device
        optimizer = self.optimizer
        criterion = self.criterion
        scheduler = self.scheduler
        train_loader = self.train_loader
        test_loader = self.test_loader

        print('%12s %12s %12s %12s %12s %12s %12s %12s %12s' % ('epoch', 'lr', 'train_time',
                                                                'train_loss', 'train_acc', 'test_time', 'test_loss', 'test_acc', 'total_time'))

        scaler = torch.cuda.amp.GradScaler(enabled=self.amp_enabled)

        time_total = 0
        for epoch in range(epochs):

            t0 = time.time()
            net = net.train()
            losses = 0
            corrects = 0
            for i, (x, y) in enumerate(train_loader):
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)

                with torch.cuda.amp.autocast(enabled=self.amp_enabled):
                    out = net(x)
                    loss = criterion(out, y)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                scheduler.step()

                losses += loss.detach()
                correct = torch.sum(torch.argmax(out, axis=1) == y)
                corrects += correct.detach()
            loss_train = losses / len(train_loader)
            acc_train = corrects / len(train_loader.dataset)
            t1 = time.time()
            time_train = t1 - t0

            t0 = time.time()
            net = net.eval()
            losses = 0
            corrects = 0
            for i, (x, y) in enumerate(test_loader):
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                with torch.no_grad():
                    out = net(x)
                    loss = criterion(out, y)

                    losses += loss.detach()

                    correct = torch.sum(torch.argmax(out, axis=1) == y)
                    corrects += correct.detach()
            loss_test = losses / len(train_loader)
            acc_test = corrects / len(test_loader.dataset)
            t1 = time.time()
            time_test = t1 - t0

            time_total += (time_train + time_test)

            log = '%12d %12.4f %12.4f %12.4f %12.4f %12.4f %12.4f %12.4f %12.4f' % (
                epoch + 1, scheduler.get_last_lr()[0], time_train, loss_train, acc_train, time_test, loss_test, acc_test, time_total)

            print(log)

        log = '| %12.4f | %12.4f | %12.4f |' % (
            time_total, acc_train, acc_test)
        print()
        print(log)


if __name__ == '__main__':
    t = Trainer()
    t.train()
