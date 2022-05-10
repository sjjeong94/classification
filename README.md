# Classification
This is a PyTorch implementation of classification on CIFAR10/CIFAR100 dataset

<br>

## CIFAR10 Experiments
### Eliminating Bottlenecks
| Case  | Description         | Duration (s) | Train Acc. | Test Acc. |
| ----- | ------------------- | ------------ | ---------- | --------- |
| 001   | baseline            | 761.4502     | 0.9822     | 0.9331    |
| 002   | amp                 | 648.1500     | 0.9825     | 0.9300    |
| 003   | remove .items()     | 604.4669     | 0.9825     | 0.9300    |
| 004   | Albumentations      | 330.7161     | 0.9907     | 0.9331    |
| 004-1 | Windows -> Linux    | 287.2552     | 0.9904     | 0.9289    |
| 005   | num_workers = 1     | 210.5783     | 0.9906     | 0.9301    |
| 006   | pin_memory = True   | 201.7149     | 0.9906     | 0.9301    |
| 008   | non_blocking = True | 194.6007     | 0.9906     | 0.9301    |
| 009   | set_to_none = True  | 190.4418     | 0.9906     | 0.9301    |
| 009-1 | batch_size -> 128   | 178.5039     | 0.9929     | 0.9288    |


### Batch Size & Weight Decay (Epochs=24)
| Case    | Description        | Duration (s) | Train Acc. | Test Acc. |
| ------- | ------------------ | ------------ | ---------- | --------- |
| 006     | bs=100, decay=5e-4 | 201.7149     | 0.9906     | 0.9301    |
| 006-1   | bs=200, decay=5e-4 | 177.5800     | 0.9959     | 0.9265    |
| 006-2   | bs=500, decay=5e-4 | 164.9722     | 0.9936     | 0.9164    |
| 006-2-1 | bs=500, decay=1e-3 | 165.4843     | 0.9942     | 0.9226    |
| 006-2-2 | bs=500, decay=2e-3 | 165.3027     | 0.9938     | 0.9267    |
| 006-2-3 | bs=500, decay=5e-3 | 164.4034     | 0.9784     | 0.9302    |


### Epochs
| Case      | Description | Duration (s) | Train Acc. | Test Acc. |
| --------- | ----------- | ------------ | ---------- | --------- |
| 006-2-3   | Epochs=24   | 164.4034     | 0.9784     | 0.9302    |
| 006-2-3-1 | Epochs=50   | 345.9947     | 0.9941     | 0.9387    |
| 006-2-3-2 | Epochs=100  | 681.2429     | 0.9979     | 0.9395    |
| 006-2-3-3 | Epochs=200  | 1394.9256    | 0.9991     | 0.9394    |

### Benchmark 
- bs=100, decay=5e-4, Epochs=100

| Case      | Description | Duration (s) | Train Acc. | Test Acc. |
| --------- | ----------- | ------------ | ---------- | --------- |
| 006-0-0-2 |             | 846.2451     | 0.9998     | 0.9451    |
| 007       | cutout      | 834.1722     | 0.9981     | 0.9470    |
| 010       | add layers  | 1216.7310    | 0.9989     | 0.9504    |

<br>

## CIFAR100 Experiments

| Case      | Description | Duration (s) | Train Acc. | Test Acc. |
| --------- | ----------- | ------------ | ---------- | --------- |
| 006       |             | 205.4771     | 0.9791     | 0.7428    |
| 006-0-0-2 |             | 833.5847     | 0.9994     | 0.7608    |
| 007       |             | 835.2124     | 0.9983     | 0.7627    |
| 010       |             | 1230.7502    | 0.9989     | 0.7713    |

<br>

## TODO
- AutoAugment