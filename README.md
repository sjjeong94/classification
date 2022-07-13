# Classification
This is a PyTorch implementation of classification on [CIFAR dataset](https://www.cs.toronto.edu/~kriz/cifar.html)

| Dataset  | Epochs | Batch Size | Time (s) | Train Acc. | Test Acc. |
| -------- | ------ | ---------- | -------- | ---------- | --------- |
| CIFAR10  | 24     | 128        | 179.09   | 99.09      | 93.21     |
| CIFAR10  | 100    | 128        | 740.42   | 99.98      | 94.39     |
| CIFAR100 | 24     | 128        | 178.40   | 98.33      | 73.85     |
| CIFAR100 | 100    | 128        | 742.26   | 99.95      | 75.87     |

## Training

~~~bash
# CIFAR10
python classification.py

# CIFAR100
python classification.py --dataset=CIFAR100
~~~

### License

MIT
