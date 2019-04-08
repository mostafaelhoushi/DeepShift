# DeepShift
## Towards a Multiplication-Less Neural Network

This is a project that aims to replace multiplications in a neural networks with bitwise shift (and sign change).

### Results So Far
#### Converting all Layers
Converting all `Conv2D` layers to `Conv2DShift` and all `Dense` layers to `DenseShift`:

| Model | Dataset | Original Version | DeepShift Version | 
|-------| ------- | -------------------------- | ----------------------------- |
| ResNet20 | CIFAR10 | 92.16% | 77.02% |
| ResNet32 | CIFAR10 | 92.46% | 85.54% |
| ResNet44 | CIFAR10 | 92.50% | 85.34% |
| ResNet50 | CIFAR10 | N/A | 86.23% |
| ResNet56 | CIFAR10 | 92.71% |  86.30% |

#### Converting Some Layers
Converting only the last `N_shift` convolution layers (as well as the last fully connected layer) to shift layers.

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
git clone http://rnd-gitlab-ca.huawei.com/Scope/Incubator/ -b DeepShift DeepShift
```
2. Install required packages:
```
pip install keras tensorflow tensorflow-gpu numpy
```
3. cd into `keras` directroy:
```
cd keras
```
4. Run the MNIST test:
```
python mnist_deepshift.py
```
5. Run the ResNet test. You can check for various options to pass:
```
python cifar10_resshift.py --help
```
and then run with the script with default options or pass the options you prefer:
```
python cifar10_resshift.py
```
