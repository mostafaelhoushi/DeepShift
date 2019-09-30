#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <vector>

// CUDA forward declarations

void linear_shift_cuda(
    torch::Tensor& input,
    torch::Tensor& shift,
    torch::Tensor& sign,
    torch::Tensor& bias,
    torch::Tensor& output);
void conv2d_shift_cuda(
    torch::Tensor& input,
    torch::Tensor& shift,
    torch::Tensor& sign,
    torch::Tensor& bias,
    torch::Tensor& output,
    torch::IntArrayRef strides,
    torch::IntArrayRef padding);
    
void GEMM_CUDA(
    torch::Tensor& input,
    torch::Tensor& shift,
    torch::Tensor& sign,
    torch::Tensor& bias,
    torch::Tensor& output,
    torch::IntArrayRef strides,
    torch::IntArrayRef padding);
// C++ interface

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

void linear_shift(
    torch::Tensor& input,
    torch::Tensor& shift,
    torch::Tensor& sign,
    torch::Tensor& bias,
    torch::Tensor& output)
{

    CHECK_INPUT(input);
    CHECK_INPUT(shift);
    CHECK_INPUT(sign);
    CHECK_INPUT(bias);
    linear_shift_cuda(input, shift, sign, bias, output);
}

void conv2d_shift(
    torch::Tensor& input,
    torch::Tensor& shift,
    torch::Tensor& sign,
    torch::Tensor& bias,
    torch::Tensor& output,
    torch::IntArrayRef strides,
    torch::IntArrayRef padding)
{
    CHECK_INPUT(input);
    CHECK_INPUT(shift);
    CHECK_INPUT(sign);
    CHECK_INPUT(bias);
    
   
    // printf("here\n");
    conv2d_shift_cuda(input, shift, sign, bias, output,strides ,padding );
}

void GEMM(
    torch::Tensor& input,
    torch::Tensor& shift,
    torch::Tensor& sign,
    torch::Tensor& bias,
    torch::Tensor& output,
    torch::IntArrayRef strides,
    torch::IntArrayRef padding)
{
    CHECK_INPUT(input);
    CHECK_INPUT(shift);
    CHECK_INPUT(sign);
    CHECK_INPUT(bias);
    
   
    // printf("here\n");
    GEMM_CUDA(input, shift, sign, bias, output,strides ,padding );
}


PYBIND11_MODULE(shift_cuda_kernel, m) {
    m.def("linear_shift", &linear_shift, "linear shift kernel(CUDA)");
    m.def("conv2d_shift", &conv2d_shift, "conv2d shift kernel(CUDA)");
    m.def("GEMM", &GEMM, "GEMM kernel(CUDA)");
}