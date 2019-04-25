# DeepShift
## Towards a Multiplication-Less Neural Network

This is a project that aims to replace multiplications in a neural networks with bitwise shift (and sign change).

### Results So Far
#### Converting all Layers
Converting all `Conv2D` layers to `Conv2DShift` and all `Dense` layers to `DenseShift`:

| Model | Dataset | Reported Original Version | DeepShift Version | 
|-------| ------- | -------------------------- | ----------------------------- |
| ResNet20 | CIFAR10 | 92.16% | 53.06% |
| ResNet32 | CIFAR10 | 92.46% | 58.21% |
| ResNet44 | CIFAR10 | 92.50% | 61.02% |
| ResNet50 | CIFAR10 | N/A | 61.97%  |
| ResNet56 | CIFAR10 | 92.71% |  63.19% |
| ResNet110 | CIFAR10 | 92.65% | 67.86% |
| MobileNet | CIFAR10 | 78.05% | 57.37% |


#### Converting Some Layers
Converting only the last `N_shift` convolution layers (as well as the last fully connected layer) to shift layers.

**ResNet50 on CIFAR10**:

| # Conv2D Layers | **# Conv2DShift Layers** | Training Accuracy | **Validation Accuracy** |
| --------------- | -------------------- | --------------------- | ------------------- |
| 21 | 0 |  |  |
| 18 | 3 | 90.9% | 88.56% |
| 17 | 4 | 90.6% | 88.05% |
| 13 | 8 | 88.0% | 84.62% |
| 0 | 21 |  | 53.06% |


**ResNet50 on CIFAR10**:

| # Conv2D Layers | **# Conv2DShift Layers** | Training Accuracy | **Validation Accuracy** |
| --------------- | -------------------- | --------------------- | ------------------- |
| 51 | 0 |  | 91.93% |
| 39 | 12 |  | 92.09% |
| 29 | 22 |  | 87.02% |
| 19 | 32 |  | 82.81% |
| 9 | 42 |  | 78.42% |
| 0 | 51 |  | 61.97% |

**MobileNet on CIFAR10**:

| # Conv2D Layers | # DWConv2D Layers | **# Conv2DShift Layers** | **# DWConv2DShift Layers** | Training Accuracy | **Validation Accuracy** |
| --------------- | ----------------- | -------------------- | ---------------------- | ----------------- | ------------------- |
| 16 | 13 | **0** | **0** | 99.7% | **78.05%** |
| 13 | 13 | **3** | **0** | 99.7% | **79.11%** |
| 13 | 12 | **3** | **1** | 99.8% | **78.34%** |
| 11 | 10 | **5** | **3** | 99.8% | **79.03%** |
| 10 | 13 | **6** | **0** | 99.7% | **78.15%** |
| 9 | 8 | **7** | **5** | 99.6% | **77.19%** |
| 7 | 13 | **9** | **0** | 99.4% | **76.69%** |
| 7 | 6 | **9** | **7** | 99.4% | **75.11%** |
| 5 | 4 | **11** | **9** | 93.3% | **70.60%** |
| 4 | 13 | **12** | **0** | 92.5% | **73.23%** |
| 2 | 2 | **14** | **11** | 75.9% | **67.04%** |
| 0 | 13 | **16** | **0** | 92.5% | **70.34%** |
| 0 | 0 | **16** | **13** | 63.5% | **57.37%** |

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
1. Clone the repo:
```
git clone http://rnd-gitlab-ca-y.huawei.com/mindspore/deepshift/ DeepShift
```
2. Change directory
```
cd DeepShift
```
3. Create virtual environment: 
```
virtualenv venv
```
4. (Needs to be done every time you run code) Source the environment:
```
source venv/bin/activate
```
5. Install required packages and build the spfpm package for fixed point
```
pip install -r requirements.txt
cd spfpm
make
```
6. cd into `keras` directroy:
```
cd keras
```
7. Run the MNIST test:
```
python mnist_deepshift.py
```
8. Run the ResNet test. You can check for various options to pass:
```
python cifar10_resshift.py --help
```
and then run with the script with default options or pass the options you prefer:
```
python cifar10_resshift.py
```
