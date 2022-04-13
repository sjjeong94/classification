# Classification
This is a PyTorch implementation of classification on MNIST/CIFAR10/CIFAR100 dataset

---

## CIFAR10 Experiment
| Case  | Description      | Training Time (s) | Train Accuracy | Test Accuracy |
| ----- | ---------------- | ----------------- | -------------- | ------------- |
| 001   | baseline         | 761.4502          | 0.9822         | 0.9331        |
| 002   | amp              | 648.1500          | 0.9825         | 0.9300        |
| 003   | remove .items()  | 604.4669          | 0.9825         | 0.9300        |
| 004   | Albumentations   | 330.7161          | 0.9907         | 0.9331        |
| 004-1 | Windows -> Linux | 287.2552          | 0.9904         | 0.9289        |
| 005   | num_workers = 1  | 210.5783          | 0.9906         | 0.9301        |
