#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <vector>

// CUDA forward declarations

torch::Tensor linear_shift_cuda(
    torch::Tensor input,
    torch::Tensor shift,
    torch::Tensor sign,
    torch::Tensor bias);

// C++ interface

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor linear_shift(
    torch::Tensor input,
    torch::Tensor shift,
    torch::Tensor sign,
    torch::Tensor bias)
{
    CHECK_INPUT(input);
    CHECK_INPUT(shift);
    CHECK_INPUT(sign);
    CHECK_INPUT(bias);
    return linear_shift_cuda(input, shift, sign, bias);
}

PYBIND11_MODULE(shift_cuda_kernel, m) {
    m.def("linear_shift", &linear_shift, "linear shift kernel(CUDA)");
}