# Classification
This is a PyTorch implementation of classification on [CIFAR dataset](https://www.cs.toronto.edu/~kriz/cifar.html)

| Dataset  | Epochs | Batch Size | Time (s) | Train Acc. | Test Acc. |
| -------- | ------ | ---------- | -------- | ---------- | --------- |
| CIFAR10  | 24     | 128        | 178.2969 | 0.9747     | 0.9323    |
| CIFAR100 | 24     | 128        | 177.8242 | 0.9516     | 0.7373    |

<br>

## Training

~~~bash
# CIFAR10
python classification.py

# CIFAR100
python classification.py --dataset=CIFAR100
~~~

### License

MIT
