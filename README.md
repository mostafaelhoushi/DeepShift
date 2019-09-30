# DeepShift
## Towards a Multiplication-Less Neural Network

This is a project that aims to replace multiplications in a neural networks with bitwise shift and sign change.

The results below are using PyTorch. Results using Keras are slightly different and its implementation still needs to be finalized.

### Results So Far
Converting all `Conv2D` layers to `Conv2DShift` and all `Dense` layers to `DenseShift`.


| Model | Dataset | Original Version | DeepShift Version<br>(Train from Scratch) | DeepShift Version<br>(Convert Original Weights) | DeepShift Version<br>(Convert Original Weights<br>+ Train) 
|-------| ------- | -------------------------- | ----------------------------- | ----------------------------- | ----------------------------- |
| Simple FC Model | MNIST | 93.59% | 78.55% | 90.19% | 93.78% |
| Simple Conv Model | MNIST | 98.91% | 85.38% | 98.41% | 98.98% |
| ResNet20 | CIFAR10 | 91.73% | 47.59% | 83.66% | 89.32% |
| ResNet32 | CIFAR10 | 92.63% | 52.92% | 85.84% | 92.16% |
| ResNet44 | CIFAR10 | 93.10% | 59.11% | 87.90% | 92.74% |
| ResNet56 | CIFAR10 | 93.39% | 61.62% | 91.03% | 93.46% |
| ResNet110 | CIFAR10 | 93.68% | 67.17% | 90.81% | 93.68% | 
| ResNet1202 | CIFAR10 | 93.82% | TBD | 91.22% | 93.63% |


| Model | Dataset | Original Version (Top1/Top5) | DeepShift Version<br>(Convert Original Weights) | DeepShift Version<br>(Convert Original Weights<br>+ Train) |
| ----- | ------- | ------------------------ | ---------------- | -------------------------- |
| VGG11 | ImageNet | 69.02% / 88.63% | 46.76% / 71.29% | 65.61% / 86.72% |
| VGG11-bn | ImageNet | 70.37% / 89.81% | 37.49% / 61.94% | 63.52% / 85.68% |
| VGG13 | ImageNet | 69.93% / 89.25% | 60.34% / 82.56% | 68.09% / 88.22% |
| VGG13-bn | ImageNet | 71.59% / 90.37% | 45.92% / 70.38% | 57.97% / 81.83% |
| VGG16 | ImageNet | 71.59% / 90.38% | 65.25% / 86.30% | 70.28% / 89.77% |
| VGG16-bn | ImageNet | 73.36% / 91.52% | 56.30% / 79.77% | 71.98% / 90.81% |
| VGG19 | ImageNet | 72.38% / 90.88% | 66.61% / 87.21% | 69.91%	/ 89.46% |
| VGG19-bn | ImageNet | 74.22% / 91.84% | 58.96% / 82.02% | 72.87% / 91.25% | 
| AlexNet | ImageNet | 56.52% / 79.07% | 42.99% / 67.40% | 48.81% / 73.39% |
| DenseNet121 | ImageNet | 74.43% / 91.97% | 46.40% / 71.95% | 70.41% / 89.93% |
| DenseNet161 | ImageNet | 77.14% / 93.56% | 61.97% / 84.64% | 73.34% / 91.55% |
| DenseNet169 | ImageNet | 75.60% / 92.81% | 39.24% / 63.93% | 72.84% / 91.28% |
| DenseNet201 | ImageNet | 76.90% / 93.37% | 51.84% / 75.83% | 73.83% / 91.80% |
| ResNet18 | ImageNet | 69.76% / 89.08% | 41.53% / 67.29% | 65.81% / 86.88% |
| ResNet34 | ImageNet | 73.31% / 91.42% | 56.26% / 80.22% | 70.99% / 90.13% |
| ResNet50 | ImageNet | 76.13% / 92.86% | 41.30% / 65.10% | 68.42% / 88.66% |
| ResNet101 | ImageNet | 77.37% / 93.55% | 52.59% / 76.57% | 69.21% / 88.95% |
| ResNet152 | ImageNet | 78.31% / 94.05% | 46.14% / 69.15% | 75.56% / 92.75% |
| MobileNetv2 | ImageNet | 71.81% / 90.42% | 0.10% / 0.48% | 63.10% / 85.26% |
| SqueezeNet1-0 | ImageNet | 58.09% / 80.42% | 12.56% / 29.92% | 21.71% / 44.74% |
| SqueezeNet1-1 | ImageNet | 58.18% / 80.62% | 4.01% / 12.19% | 15.50% / 35.25% |

### Getting Started
1. Clone the repo:
```
git -c lfs.url=http://ptlab01.huawei.com:31337/ clone git@rnd-gitlab-ca.huawei.com:Do4AI/DeepShift.git
```

git will prompt interactively for the username and password with which to access ptlab01.huawei.com.

2. Change directory
```
cd DeepShift
```
3. Save the LFS URL in the clone's settings:
```
git config lfs.url http://ptlab01.huawei.com:31337/
```

To guarantee that the code works on your machine, we recommend that you create a virtual environment to install the same packages that we used to develop the code.

4. Create virtual environment using Python version 3.5: 
```
virtualenv venv --python=/usr/bin/python3.5 --prompt="(DeepShift) "
```
5. (Needs to be done every time you run code) Source the environment:
```
source venv/bin/activate
```
6. Install the required packages
```
pip install -r requirements.txt
```


7. Install our CPU and CUDA kernels that perform matrix multiplication and convolution using bit-wise shifts:
```
cd cpu_kernal
python setup.py install
cd ...

cd cuda_kernel
python setup.py install
cd ...
```


8. cd into `pytorch` directroy:
```
cd pytorch
```
9. Now you can run the different scripts with different options, e.g.,
    a) Train a DeepShift simple fully-connected model on the MNIST dataset:
    ```
    python mnist.py --shift 3
    ```
    b) Train a DeepShift simple convolutional model on the MNIST dataset:
    ```
    python mnist.py --type conv --shift 3
    ```
    c) Train a DeepShift ResNet20 on the CIFAR10 dataset from scratch:
    ```
    python cifar10.py -a resnet20 --pretrained False --shift 1000 
    ```
    d) Infer a DeepShift ResNet20 model on the CIFAR10 dataset using converted pretrained weights:
    ```
    python cifar10.py -a resnet20 --pretrained True --shift 1000 --evaluate
    ```
    e) Train a DeepShift ResNet20 model on the CIFAR10 dataset starting from the original pretrained weights:
    ```
    python cifar10.py -a resnet20 --pretrained True --shift 1000
    ```
    f) Infer a DeepShift VGG19 model on the Imagenet dataset using converted pretrained weights:
    ```
    python imagenet.py -a vgg19 --pretrained True --shift 1000 --evaluate
    ```
    g) Train a DeepShift DenseNet121 model on the Imagenet dataset using converted pretrained weights for 10 epochs with learning rate 0.01:
    ```
    python imagenet.py -a densenet121 --pretrained True --shift 1000 --epochs 10 --lr 0.01
    ```

### Codewalk Through
* `pytorch`: directory containing implementation, tests, and saved models using PyTorch
    * `shift.py`: definition of `LinearShift` and `ConvShift` ops
    * `mnist.py`: example script to train and infer on MNIST dataset using simple models in both their original forms and DeepShift version.
    * `cifar10.py`: example script to train and infer on CIFAR10 dataset using various models in both their original forms and DeepShift version.
    * `imagenet.py`: example script to train and infer on Imagenet dataset using various models in both their original forms and DeepShift version.
    * `models`: directory containing saved PyTorch model files and training results
