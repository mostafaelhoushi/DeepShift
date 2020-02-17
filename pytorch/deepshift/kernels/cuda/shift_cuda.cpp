#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <vector>

// CUDA forward declarations
void COMPRESS_SIGN_SHIFT_GPU(torch::Tensor shift, torch::Tensor sign, torch::Tensor weight, 
                        int base, int bits, int out_c, int in_c, int height, int width, int row_length, int num);
void DEEP_SHIFT_CONV_GPU(torch::Tensor data_im,
                torch::Tensor shift,
                torch::Tensor bias,
                torch::Tensor output,
                torch::IntArrayRef strides, 
                torch::IntArrayRef padding, int filter_height, int filter_width, int base, int bits);
void DEEP_SHIFT_LINEAR_GPU(
    torch::Tensor input,
    torch::Tensor shift,
    torch::Tensor bias,
    torch::Tensor output,
    int base, int bits, int out_features);
// C++ interface

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

void DEEP_SHIFT_LINEAR(
    torch::Tensor input,
    torch::Tensor shift,
    torch::Tensor bias,
    torch::Tensor output,
    int base, int bits, int out_features)
{

    CHECK_INPUT(input);
    CHECK_INPUT(shift);
    CHECK_INPUT(bias);
    CHECK_INPUT(output);
    DEEP_SHIFT_LINEAR_GPU(input, shift, bias, output, base, bits, out_features);
}

void DEEP_SHIFT_CONV(torch::Tensor data_im,
        torch::Tensor shift,
        torch::Tensor bias,
        torch::Tensor output,
        torch::IntArrayRef strides, 
        torch::IntArrayRef padding, int filter_height, int filter_width, int base, int bits)
{
    CHECK_INPUT(data_im);
    CHECK_INPUT(shift);
    CHECK_INPUT(bias);
    CHECK_INPUT(output);
    DEEP_SHIFT_CONV_GPU(data_im, shift, bias, output, strides, padding, filter_height, filter_width, base,bits);
}

void COMPRESS_SIGN_SHIFT(torch::Tensor shift, torch::Tensor sign, torch::Tensor weight, int base, int bits, int out_c, int in_c, int height, int width, int row_length, int num)
{
    CHECK_INPUT(shift);
    CHECK_INPUT(sign);
    CHECK_INPUT(weight);
    COMPRESS_SIGN_SHIFT_GPU(shift,sign, weight, base,bits, out_c, in_c, height,width,row_length, num);
}


PYBIND11_MODULE(deepshift_cuda, m) {
    m.def("DEEP_SHIFT_LINEAR", &DEEP_SHIFT_LINEAR, "DEEP_SHIFT_LINEAR kernel(CUDA)");
    m.def("DEEP_SHIFT_CONV", &DEEP_SHIFT_CONV, "DEEP_SHIFT_CONV kernel(CUDA)");
    m.def("COMPRESS_SIGN_SHIFT", &COMPRESS_SIGN_SHIFT, "COMPRESS_SIGN_SHIFT kernel(CUDA)");
}
