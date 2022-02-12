import torch
import models
from utils import *


def main():
    set_seed(1234)

    learning_rate = 0.1
    weight_decay = 0.0005
    batch_size = 128
    num_epochs = 200

    dataset = 'CIFAR10'  # MNIST, FashionMNIST, CIFAR10, CIFAR100
    name = 'EfficientNetB0_001'
    dataset_root = f'./{dataset}/data'
    model_root = f'./{dataset}/model/{name}'
    result_root = f'./{dataset}/result'
    image_root = f'./{dataset}/image'

    train_loader, test_loader = get_dataset(dataset, dataset_root, batch_size)

    net = models.EfficientNetB0(10)
    device = get_device()
    optimizer = torch.optim.SGD(
        net.to(device).parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=50, gamma=0.2)

    print(net)
    print(device)
    print(optimizer)

    engine = Engine(
        name=name,
        net=net,
        device=device,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        test_loader=test_loader,
        num_epochs=num_epochs,
        model_root=model_root,
        result_root=result_root,
        image_root=image_root,
    )

    engine.train()
    engine.evaluate()
    engine.plot()


if __name__ == '__main__':
    main()
