# Classification
This is a PyTorch implementation of classification on [CIFAR dataset](https://www.cs.toronto.edu/~kriz/cifar.html)

| Dataset  | Epochs | Batch Size | Time (s) | Train Acc. | Test Acc. |
| -------- | ------ | ---------- | -------- | ---------- | --------- |
| CIFAR10  | 24     | 128        | 152.30   | 99.13      | 92.98     |
| CIFAR10  | 100    | 128        | 633.76   | 99.99      | 94.41     |
| CIFAR100 | 24     | 128        | 152.20   | 98.43      | 73.95     |
| CIFAR100 | 100    | 128        | 630.41   | 99.96      | 76.10     |

## Training

~~~bash
# CIFAR10
python classification.py

# CIFAR100
python classification.py --dataset=CIFAR100
~~~

### License

MIT
