import os
import random
import numpy as np
import matplotlib.pyplot as plt
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


def get_device():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)
    return device


def get_dataset(dataset, root, batch_size):

    if dataset == 'MNIST':
        load_dataset = datasets.MNIST
    elif dataset == 'FashionMNIST':
        load_dataset = datasets.FashionMNIST
    elif dataset == 'CIFAR10':
        load_dataset = datasets.CIFAR10
    elif dataset == 'CIFAR100':
        load_dataset = datasets.CIFAR100

    transform = transforms.Compose([
        transforms.Pad(4, padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32),
        transforms.ToTensor()
    ])

    train_dataset = load_dataset(
        root=root, train=True, transform=transform, download=True)
    test_dataset = load_dataset(
        root=root, train=False, transform=transforms.ToTensor(), download=False)

    # Data Loader
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


class Engine:
    def __init__(
        self,
        name,
        net,
        device,
        optimizer,
        train_loader,
        test_loader,
        num_epochs,
        model_root,
        result_root,
        image_root,
    ):
        self.name = name
        self.net = net
        self.device = device
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.num_epochs = num_epochs
        self.model_root = model_root
        self.result_root = result_root
        self.image_root = image_root

    def train(self):
        net = self.net.to(self.device)
        criterion = nn.CrossEntropyLoss()

        print('model_root:', self.model_root)
        os.makedirs(self.model_root, exist_ok=True)
        for epoch in range(1, self.num_epochs+1):
            net = net.to(self.device)
            net = net.train()
            losses = []
            for i, (x, y) in enumerate(self.train_loader):
                x = x.to(self.device)
                y = y.to(self.device)
                net.zero_grad()
                out = net(x)
                loss = criterion(out, y)
                loss.backward()
                self.optimizer.step()
                losses.append(loss.item())
            loss_mean = np.mean(losses)
            model_path = os.path.join(
                self.model_root, '%s_epoch%03d.pth' % (self.name, epoch))
            torch.save(net.cpu().state_dict(), model_path)
            print('epoch %4d  |  loss %9.6f  |   model_path -> %s' %
                  (epoch, loss_mean, model_path))
        self.net = net

    def evaluate(self):
        net = self.net.to(self.device)
        net = net.eval()
        criterion = nn.CrossEntropyLoss()

        models = sorted(os.listdir(self.model_root))

        losses_train = []
        losses_test = []
        accuracy_train = []
        accuracy_test = []
        with torch.no_grad():
            for model in models:
                if model[-3:] != 'pth':
                    continue
                model_path = os.path.join(self.model_root, model)
                net.load_state_dict(torch.load(model_path))
                print('model_path: ', model_path)

                loss_train = []
                corr_train = []
                for i, (x, y) in enumerate(self.train_loader):
                    x = x.to(self.device)
                    y = y.to(self.device)
                    out = net(x)
                    loss = criterion(out, y)
                    pred = torch.argmax(out, axis=1)
                    loss_train.append(loss.item())
                    corr_train.append(torch.sum(pred == y).item())
                l_train = np.mean(loss_train)
                a_train = np.sum(corr_train) / len(self.train_loader.dataset)

                loss_test = []
                corr_test = []
                for i, (x, y) in enumerate(self.test_loader):
                    x = x.to(self.device)
                    y = y.to(self.device)
                    out = net(x)
                    loss = criterion(out, y)
                    pred = torch.argmax(out, axis=1)
                    loss_test.append(loss.item())
                    corr_test.append(torch.sum(pred == y).item())
                l_test = np.mean(loss_test)
                a_test = np.sum(corr_test) / len(self.test_loader.dataset)

                print('Loss = [%f | %f] Accuracy = [%f | %f]' %
                      (l_train, l_test, a_train, a_test))

                losses_train.append(l_train)
                losses_test.append(l_test)
                accuracy_train.append(a_train)
                accuracy_test.append(a_test)

        result = np.array([losses_train, losses_test,
                           accuracy_train, accuracy_test])

        os.makedirs(self.result_root, exist_ok=True)
        result_path = os.path.join(self.result_root, '%s.npy' % self.name)
        np.save(result_path, result)
        print('result_path:', result_path)

        self.net = net
        self.result = result

    def plot(self):
        result = self.result
        epochs = np.arange(1, result.shape[1] + 1)
        plt.figure(figsize=(16, 4))
        plt.subplot(1, 2, 1)
        plt.plot(epochs, result[0], label='train')
        plt.plot(epochs, result[1], label='test')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.title(self.name)
        plt.legend()
        plt.grid()
        plt.subplot(1, 2, 2)
        plt.plot(epochs, result[2], label='train')
        plt.plot(epochs, result[3], label='test')
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.title(self.name)
        plt.legend()
        plt.grid()

        os.makedirs(self.image_root, exist_ok=True)
        image_path = os.path.join(self.image_root, '%s.png' % self.name)
        plt.savefig(image_path)
        print('image_path:', image_path)
