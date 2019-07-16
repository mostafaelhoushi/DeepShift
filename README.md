# DeepShift
## Towards a Multiplication-Less Neural Network

This is a project that aims to replace multiplications in a neural networks with bitwise shift (and sign change).

The results below are using PyTorch. Results using Keras are slightly different and its implementation still needs to be finalized.

### Results So Far
Converting all `Conv2D` layers to `Conv2DShift` and all `Dense` layers to `DenseShift`.

When converting a convolution layer or fully connected layer to a shifted layer, the following conversion is made to get the shifts that are closest in equivalent to the original weights: 
```
[kernel, bias] = original_layer.get_weights()
shift = log2(round(abs(kernel)))
sign = sign(kernel)
sign[sign==1] = 0
sign[sign==-1] = 1
shift_layer.set_weights([shift, sign, bias])
```
and then the model is re-trained for a small number of epochs.

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
| MobileNetv1 | ImageNet | 69.57% / 89.07% | 0.14% / 0.67% | 60.61% / 83.23% |
| MobileNetv2 | ImageNet | 71.81% / 90.42% | 0.10% / 0.48% | 63.10% / 85.26% |
| SqueezeNet1-0 | ImageNet | 58.09% / 80.42% | 12.56% / 29.92% | 21.71% / 44.74% |
| SqueezeNet1-1 | ImageNet | 58.18% / 80.62% | 4.01% / 12.19% | 15.50% / 35.25% |


### Codewalk Through
* `keras`: directory containing implementation, tests, and saved models using Keras
    * `shift_layer.py`: definition of `DenseShift` class that inherits from Keras' `Layer` class and modifies it to implement fully connected (a.k.a Dense) layer as multiplications with integer powers of 2 (mathematically equialvent to bitwaise shift) and  integer powers of -1 (mathematically equivalent to sign change or sign keep). 
    * `convolutional_shift.py`: definition of `Conv2DShift` class that inherits from Keras` `Conv2D` layer to implement convolution as multiplications with integer powers of 2 (mathematically equialvent to bitwaise shift) and  integer powers of -1 (mathematically equivalent to sign change or sign keep).
    * `mnist_deepshift.py`: example script to classify MNIST dataset using a small fully connected model.
    * `cifar10_resshift.py`: modify different depth versions of ResNet50 to use shift convolutions instead of multiplication convolutions.
    * `selected_models`: directory containing saved Keras model files and training results

### TODOs
Currently, both trainining and inference are actually done by multiplying by a power of 2 and by a power of -1.
So we need to:
- Implement the inference on fixed float data using bitwise shift (instead of multiplying by a power of 2) and bitwise XOR (instead of negating), and verify that its speed on CPU will be faster than multiplication.
- Implement training on fixed float data using bitwaise shift and bitwise XOR, and verify that its speed on CPU will be faster than multiplication.
- Implement an FPGA prototype to execute bitwise shift and XOR  in parallel.
- Investigate different methods or hyperparameter tuning to enhance the accuracy results.

### Running the Code
1. Clone the repo along with the model files that exist on the LFS server:
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
4. Create virtual environment: 
```
virtualenv venv --python=/usr/bin/python3.6 --prompt="(DeepShift) "
```
5. (Needs to be done every time you run code) Source the environment:
```
source venv/bin/activate
```
6. Install required packages and build the spfpm package for fixed point
```
pip install -r requirements.txt
cd spfpm
make
```
7. cd into `keras` directroy:
```
cd keras
```
8. Run the MNIST test:
```
python mnist_deepshift.py
```
9. Run the ResNet test. You can check for various options to pass:
```
python cifar10_resshift.py --help
```
and then run with the script with default options or pass the options you prefer:
```
python cifar10_resshift.py
```
