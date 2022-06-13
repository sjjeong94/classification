# Classification
This is a PyTorch implementation of classification on [CIFAR dataset](https://www.cs.toronto.edu/~kriz/cifar.html)

| Dataset  | Epochs | Batch Size | Time (s) | Train Acc. | Test Acc. |
| -------- | ------ | ---------- | -------- | ---------- | --------- |
| CIFAR10  | 24     | 128        | 178.39   | 97.47      | 93.23     |
| CIFAR10  | 100    | 128        | 739.43   | 99.72      | 94.61     |
| CIFAR100 | 24     | 128        | 178.03   | 95.16      | 73.73     |
| CIFAR100 | 100    | 128        | 739.23   | 99.78      | 76.53     |

## Training

~~~bash
# CIFAR10
python classification.py

# CIFAR100
python classification.py --dataset=CIFAR100
~~~

### License

MIT
