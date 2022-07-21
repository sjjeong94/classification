# Classification
This is a PyTorch implementation of classification on [CIFAR dataset](https://www.cs.toronto.edu/~kriz/cifar.html)

| Dataset  | Epochs | Batch Size | Time (s) | Train Acc. | Test Acc. |
| -------- | ------ | ---------- | -------- | ---------- | --------- |
| CIFAR10  | 25     | 128        | 159.23   | 99.23      | 93.31     |
| CIFAR10  | 100    | 128        | 633.76   | 99.99      | 94.41     |
| CIFAR100 | 25     | 128        | 159.30   | 98.68      | 73.96     |
| CIFAR100 | 100    | 128        | 630.41   | 99.96      | 76.10     |

## Training

~~~bash
# CIFAR10
python train.py

# CIFAR100
python train.py --dataset=CIFAR100
~~~

### License

MIT
