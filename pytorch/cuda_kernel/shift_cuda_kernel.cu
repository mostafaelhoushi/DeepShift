
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>
template <typename scalar_t>
__global__ void linear_shift_cuda_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ shift,
    const scalar_t* __restrict__ sign,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ output,
    size_t input_features) 
{
   
    
    for(int i = 0; i < input_features; i++){
        auto x = input[blockIdx.x * input_features + i];
        auto s = shift[threadIdx.x * input_features + i];
        auto y = output[blockIdx.x * blockDim.x + threadIdx.x];
        if((bool)sign[threadIdx.x * input_features + i]){
            if(s >= 0){
                y -= (x << s);
            }
            else{
                y -= (x >> (-s));
            }
        }
        else{
            if(s >= 0){
                y += (x << s);
            }
            else{
                y += (x >> (-s));
            }

        }
        output[blockIdx.x * blockDim.x + threadIdx.x]=y;
        // if(blockIdx.x * blockDim.x + threadIdx.x == 1){
        //     printf("x: %d, s: %d, sign: %d, y: %d\n", x,s,sign[threadIdx.x * input_features + i],y);
        // }
    }
    output[blockIdx.x * blockDim.x + threadIdx.x] += bias[threadIdx.x];
}


torch::Tensor linear_shift_cuda(
    torch::Tensor input,
    torch::Tensor shift,
    torch::Tensor sign,
    torch::Tensor bias)
{

    auto output =  torch::zeros({input.size(0),shift.size(0)}, torch::dtype(torch::kInt32));
    output = output.to(at::kCUDA);
    const int block = input.size(0);
    const int threads = shift.size(0);
    // std::cout<<input.type()<<std::endl;
    // std::cout<<shift.type()<<std::endl;
    // std::cout<<sign.type()<<std::endl;
    // std::cout<<bias.type()<<std::endl;
    // std::cout<<output.type()<<std::endl;
    AT_DISPATCH_INTEGRAL_TYPES(input.type(), "test cuda", ([&] {
        linear_shift_cuda_kernel<scalar_t><<<block, threads>>>(
            input.data<scalar_t>(),
            shift.data<scalar_t>(),
            sign.data<scalar_t>(),
            bias.data<scalar_t>(),
            output.data<scalar_t>(),
            (int)input.size(1));
      }));

      return output;
}



