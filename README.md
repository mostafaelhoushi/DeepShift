# DeepShift
## Towards a Multiplication-Less Neural Network

This is a project that aims to replace multiplications in a neural networks with bitwise shift (and sign change).

### Results So Far
#### Converting all Layers
Converting all `Conv2D` layers to `Conv2DShift` and all `Dense` layers to `DenseShift`:

| Model | Dataset | Original Version | DeepShift Version | 
|-------| ------- | -------------------------- | ----------------------------- |
| MobileNet | CIFAR10 | 78.05% | TBD |
| ResNet20 | CIFAR10 | 92.16% | <del>77.02%<del> |
| ResNet32 | CIFAR10 | 92.46% | <del>85.54%<del> |
| ResNet44 | CIFAR10 | 92.50% | <del>85.34%<del> |
| ResNet50 | CIFAR10 | N/A | 61.97% <del>86.23%<del> |
| ResNet56 | CIFAR10 | 92.71% |  63.19% <del>86.30%<del> |
| ResNet110 | CIFAR10 | 92.65% | 67.86% |

#### Converting Some Layers
Converting only the last `N_shift` convolution layers (as well as the last fully connected layer) to shift layers.

**ResNet50 on CIFAR10**:

| # Conv2D Layers | **# Conv2DShift Layers** | **Validation Accuracy** |
| --------------- | -------------------- | ------------------- |
| 51 | 0 | 91.93% |
| 39 | 12 | 92.090% <del>92.14%<del> |
| 29 | 22 | 87.02% <del>91.65%<del> |
| 19 | 32 | 82.81% <del>88.83%<del> |
| 9 | 42 | 78.42% <del>88.12%<del> |
| 0 | 51 | 61.97% <del>86.23%<del> |

**MobileNet on CIFAR10**:

| # Conv2D Layers | # DWConv2D Layers | **# Conv2DShift Layers** | **# DWConv2DShift Layers** | Training Accuracy | **Validation Accuracy** |
| --------------- | ----------------- | -------------------- | ---------------------- | ----------------- | ------------------- |
| 16 | 14 | **0** | **0** | 99.7% | **78.05%** |
| 13 | 14 | **3** | **0** | 99.7% | **79.11%** |
| 10 | 14 | **6** | **0** | 99.7% | **78.15%** |
| 7 | 14 | **9** | **0** | 99.4% | **76.69%** |
| 4 | 14 | **12** | **0** | 92.5% | **73.23%** |


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
