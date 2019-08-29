
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ctime>
#include <vector>
template <typename scalar_t>
__global__ void linear_shift_cuda_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ shift,
    const scalar_t* __restrict__ sign,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ output,
    size_t input_features,
    int out_height,
    int out_width) 
{
    if(blockIdx.x * blockDim.x + threadIdx.x < out_height * out_width){
        int idx_h = (blockIdx.x * blockDim.x + threadIdx.x) / out_width;
        int idx_w = (blockIdx.x * blockDim.x + threadIdx.x) % out_width;
        
        for(int i = 0; i < input_features; i++){
            auto x = input[idx_h * input_features + i];
            auto s = shift[idx_w * input_features + i];
            auto y = output[blockIdx.x * blockDim.x + threadIdx.x];
            if((bool)sign[idx_w * input_features + i]){
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
            // if(blockIdx.x * blockDim.x + threadIdx.x == 512){
            //     printf("x: %d, s: %d, sign: %d, y: %d\n", x,s,sign[threadIdx.x * input_features + i],y);
            //     // printf("iidx_w: %d\n",idx_w);
            // }
        }
        // if(blockIdx.x * blockDim.x + threadIdx.x == 513){
        //     // printf("x: %d, s: %d, sign: %d, y: %d\n", x,s,sign[threadIdx.x * input_features + i],y);
        //     printf("iidx_w: %d, h: %d\n",idx_w,idx_h);
        // }
        output[blockIdx.x * blockDim.x + threadIdx.x] += bias[idx_w];
    }
    
}

template <typename scalar_t>
__global__ void conv2d_shift_cuda_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ shift,
    const scalar_t* __restrict__ sign,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ output,
    int filter_height,
    int filter_width,
    int input_features,
    int out_height,
    int out_width,
    int strides_h,
    int strides_w,
    int oc,
    int in_width,
    int in_height)
{
    
    
    int idx = blockIdx.y * 1024 + threadIdx.x;
    // if(idx == 0 && blockIdx.x == 0){
    //     printf("in kernel\n");
    // }
    if(idx < out_height * out_width * oc){
        int batch = blockIdx.x;
        int out_channel = idx / (out_height * out_width);
        int h = (idx % (out_height * out_width)) / out_width;
        int w = (idx % (out_height * out_width)) % out_width;

        for(int i = 0; i < filter_height; i++){
            for(int j = 0 ; j < filter_width; j++){
                for(int k = 0 ; k < input_features; k++){
                    // auto s = shift[out_channel][k][i][j];
                    auto s = shift[j + i * filter_width 
                                    + k * filter_height * filter_width + 
                                    out_channel * filter_height * filter_width * input_features];
                    // auto y = output[batch][out_channel][h][w];
                    auto y = output[w + h * out_width + out_channel * out_width * out_height 
                                    + batch * oc * out_width * out_height];
                    // auto x = input[batch][k][i + strides_h * h][j + strides_w * w];
                    auto x = input[j + strides_w * w + (i + strides_h * h) * in_width 
                                    + k * in_width * in_height 
                                    + batch * in_width * in_height * input_features];
                    if((bool)sign[j + i * filter_width 
                                + k * filter_height * filter_width + 
                                out_channel * filter_height * filter_width * input_features]){
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
                    output[w + h * out_width + out_channel * out_width * out_height 
                    + batch * oc * out_width * out_height] = y;

                }
            }
        }

        output[w + h * out_width + out_channel * out_width * out_height 
                + batch * oc * out_width * out_height] += bias[out_channel];
    }
    
}

// template <typename scalar_t>
// __global__ void im2col(
//     const torch::PackedTensorAccessor<int,4,torch::RestrictPtrTraits,size_t> im,
//     torch::PackedTensorAccessor<int,3,torch::RestrictPtrTraits,size_t> col,
//     int filter_height,
//     int filter_width,
//     int input_features,
//     int out_height,
//     int out_width,
//     int strides_h,
//     int strides_w)
// {
//     int batch = blockIdx.x;
//     int idx = blockIdx.y * 1024 + threadIdx.x;
//     int k = filter_height * filter_width * input_features;
//     int pitch = out_height * out_width;
//     int num_pitch = idx / k;
//     int num_oc = idx % k;
//     int inc_height = (num_pitch / out_width) * strides_h;
//     int inc_width = (num_pitch % out_width) * strides_w;
//     int filter_area = filter_height * filter_width;
//     int feature = num_oc / filter_area;
//     int height = (num_oc % filter_area) / filter_width;
//     int width = (num_oc % filter_area) % filter_width;
//     col[batch][num_pitch][num_oc] = im[batch][feature][height + inc_height][width + inc_width];
//     // if(blockIdx.x == 0 && blockIdx.y=0 && threadIdx.x=0){
//     //     printf("data: %d, %d, %d, %d\n", im[blockIdx.x][feature][height][width], feature,height,width);
//     // }
// }

torch::Tensor linear_shift_cuda(
    torch::Tensor& input,
    torch::Tensor& shift,
    torch::Tensor& sign,
    torch::Tensor& bias)
{
    
    auto output =  torch::zeros({input.size(0),shift.size(0)}, torch::dtype(torch::kInt32));
    output = output.to(at::kCUDA);
    const int block = (input.size(0) * shift.size(0) + 1024 -1) / 1024;
    const int threads = 1024;
    // std::cout<<input.type()<<std::endl;
    // std::cout<<shift.type()<<std::endl;
    // std::cout<<sign.type()<<std::endl;
    // std::cout<<bias.type()<<std::endl;
    // std::cout<<output.type()<<std::endl;
    
    AT_DISPATCH_INTEGRAL_TYPES(input.type(), "linear shift kernel", ([&] {
        linear_shift_cuda_kernel<scalar_t><<<block, threads>>>(
            input.data<scalar_t>(),
            shift.data<scalar_t>(),
            sign.data<scalar_t>(),
            bias.data<scalar_t>(),
            output.data<scalar_t>(),
            (int)input.size(1),
            input.size(0),
            shift.size(0));
      }));

      return output;
}


torch::Tensor conv2d_shift_cuda(
    torch::Tensor& input,
    torch::Tensor& shift,
    torch::Tensor& sign,
    torch::Tensor& bias,
    torch::IntArrayRef strides,
    torch::IntArrayRef padding)
{
    
    
    int strides_h;
    int strides_w;
    if(strides.size() ==1){
        strides_h = strides[0];
        strides_w = strides[0];
    }
    else{
        strides_h = strides[0];
        strides_w = strides[1];
    }
    
    int out_height = (input.size(2) -shift.size(2)) / strides_h +1;
    int out_width = (input.size(3) - shift.size(3)) / strides_w +1;
    
    auto output = torch::zeros({input.size(0),shift.size(0),out_height,out_width}, torch::dtype(torch::kInt32));

    output = output.to(at::kCUDA);
    
    int temp = (out_height * out_width * shift.size(0)  + 1024 - 1) / 1024;
    const dim3 block(input.size(0), temp);
    const int threads = 1024;
    // start = std::clock();
    // std::cout<<"hello"<<std::endl;

    AT_DISPATCH_INTEGRAL_TYPES(input.type(), "conv2d cuda", ([&] {
        conv2d_shift_cuda_kernel<scalar_t><<<block, threads>>>(
            input.data<scalar_t>(),
            shift.data<scalar_t>(),
            sign.data<scalar_t>(),
            bias.data<scalar_t>(),
            output.data<scalar_t>(),
            shift.size(2),
            shift.size(3),
            input.size(1),
            out_height,
            out_width,
            strides_h,
            strides_w,
            shift.size(0),
            input.size(3),
            input.size(2));
    }));
    
    // duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;

    // std::cout<<"One batch use "<< duration <<'\n';

    return output;
}






