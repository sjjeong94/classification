import torch
import models
from utils import *

# TODO: make class


def main():
    seed = 1234
    num_epochs = 100
    learning_rate = 0.0003
    weight_decay = 0
    batch_size = 100
    dataset = 'CIFAR10'  # MNIST, FashionMNIST, CIFAR10, CIFAR100
    name = 'MobileNetV2_001'
    net = models.MobileNetV2(10)
    print(net)
    set_seed(seed)
    device = get_device()
    net = net.to(device)
    optimizer = torch.optim.Adam(
        net.parameters(), lr=learning_rate, weight_decay=weight_decay)

    dataset_root = f'./{dataset}/data'
    model_root = f'./{dataset}/model/{name}'
    result_root = f'./{dataset}/result'
    image_root = f'./{dataset}/image'

    train_loader, test_loader = get_dataset(dataset, dataset_root, batch_size)

    net = train(
        name=name,
        net=net,
        device=device,
        train_loader=train_loader,
        optimizer=optimizer,
        model_root=model_root,
        num_epochs=num_epochs,
    )

    result = evaluate(
        name=name,
        net=net,
        device=device,
        train_loader=train_loader,
        test_loader=test_loader,
        model_root=model_root,
        result_root=result_root,
    )

    plot(
        result=result,
        name=name,
        image_root=image_root
    )


if __name__ == '__main__':
    main()
