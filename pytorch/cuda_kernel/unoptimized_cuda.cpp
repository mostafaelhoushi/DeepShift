#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <vector>

// CUDA forward declarations

void UNOPTIMIZED_LINEAR_GPU(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor output);
void UNOPTIMIZED_CONV_GPU(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor output,
    torch::IntArrayRef strides,
    torch::IntArrayRef padding);
// C++ interface

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

void UNOPTIMIZED_LINEAR(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor output)
{

    CHECK_INPUT(input);
    CHECK_INPUT(weight);
    CHECK_INPUT(bias);
    UNOPTIMIZED_LINEAR_GPU(input, weight, bias, output);
}

void UNOPTIMIZED_CONV(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor output,
    torch::IntArrayRef strides,
    torch::IntArrayRef padding)
{
    CHECK_INPUT(input);
    CHECK_INPUT(weight);
    CHECK_INPUT(bias);
    
    UNOPTIMIZED_CONV_GPU(input, weight, bias, output,strides ,padding );
}

PYBIND11_MODULE(unoptimized_cuda_kernel, m) {
    m.def("UNOPTIMIZED_LINEAR", &UNOPTIMIZED_LINEAR, "UNOPTIMIZED_LINEAR kernel(CUDA)");
    m.def("UNOPTIMIZED_CONV", &UNOPTIMIZED_CONV, "UNOPTIMIZED_CONV kernel(CUDA)");
}
