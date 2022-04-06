import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms


def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def train(epochs, net, train_loader, test_loader, device, criterion, optimizer, scheduler):
    print('%12s %12s %12s %12s %12s %12s %12s %12s %12s' % ('epoch', 'lr', 'train_time',
          'train_loss', 'train_acc', 'test_time', 'test_loss', 'test_acc', 'total_time'))

    time_total = 0
    for epoch in range(epochs):

        t0 = time.time()
        net = net.train()
        losses = []
        corrects = 0
        for i, (x, y) in enumerate(train_loader):
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            out = net(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            scheduler.step()
            losses.append(loss.item())
            correct = torch.sum(torch.argmax(out, axis=1) == y)
            corrects += correct.item()
        loss_train = np.mean(losses)
        acc_train = corrects / len(train_loader.dataset)
        t1 = time.time()
        time_train = t1 - t0

        t0 = time.time()
        net = net.eval()
        losses = []
        corrects = 0
        for i, (x, y) in enumerate(test_loader):
            x = x.to(device)
            y = y.to(device)
            with torch.no_grad():
                out = net(x)
                loss = criterion(out, y)
                losses.append(loss.item())

                correct = torch.sum(torch.argmax(out, axis=1) == y)
                corrects += correct.item()
        loss_test = np.mean(losses)
        acc_test = corrects / len(test_loader.dataset)
        t1 = time.time()
        time_test = t1 - t0

        time_total += (time_train + time_test)

        log = '%12d %12.4f %12.4f %12.4f %12.4f %12.4f %12.4f %12.4f %12.4f' % (
            epoch + 1, scheduler.get_last_lr()[0], time_train, loss_train, acc_train, time_test, loss_test, acc_test, time_total)

        print(log)


def exp_000():
    set_seed(1234)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    root = './data'
    batch_size = 100
    T_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    T_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    load_dataset = datasets.CIFAR10

    train_dataset = load_dataset(
        root=root, train=True, transform=T_train, download=True)
    test_dataset = load_dataset(
        root=root, train=False, transform=T_test, download=False)

    # Data Loader
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=batch_size, shuffle=False)

    net = nn.Sequential(
        nn.Conv2d(3, 64, 3, 1, 1, bias=False),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.Conv2d(64, 128, 3, 1, 1, bias=False),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),
        nn.Conv2d(128, 256, 3, 1, 1, bias=False),
        nn.BatchNorm2d(256),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),
        nn.Conv2d(256, 512, 3, 1, 1, bias=False),
        nn.BatchNorm2d(512),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(512, 10),
    )
    net = net.to(device)

    epochs = 20
    learning_rate = 0.1
    momentum = 0.9
    weight_decay = 5e-4
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(
    ), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                                    max_lr=learning_rate,
                                                    epochs=epochs,
                                                    steps_per_epoch=len(
                                                        train_loader),
                                                    anneal_strategy='linear')

    train(epochs, net, train_loader, test_loader,
          device, criterion, optimizer, scheduler)


if __name__ == '__main__':
    exp_000()
