# Cross-Layer Graph Knowledge Distillation for Image Recognition

This project provides source code for official implementation of Cross-Layer Graph Knowledge Distillation (CLGKD):

## Installation

### Requirements

Ubuntu 20.04 LTS

Python 3.9

CUDA 11.1

PyTorch 1.7.0

please install python packages:

```
pip install -r requirements.txt
```

## Perform experiments on CIFAR-100 dataset

### Dataset

CIFAR-100 : [download](http://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz)

unzip to the `./data` folder

### Training a teacher network
```
python train_teacher_cifar.py \
    --arch wrn_40_2_aux \
    --data [your dataset path] \
    --checkpoint-dir [your checkpoint saving path] \
```


### Training a student baseline
```
python train_baseline_cifar.py \
    --arch wrn_16_2 \
    --data [your dataset path] \
    --checkpoint-dir [your checkpoint saving path] \
```

### Training a student by CLGKD
```
python train_student_cifar.py \
    --tarch wrn_40_2_aux \
    --arch wrn_16_2_aux \
    --data [your dataset path] \
    --tcheckpoint [your pretrained teacher model path] \
    --checkpoint-dir [your checkpoint saving path] \
```

