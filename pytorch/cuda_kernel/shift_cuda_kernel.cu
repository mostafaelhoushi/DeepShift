
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ctime>
#include <vector>
// #include <ATen/native/cuda/im2col.cuh>
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
           
        }
       
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
    
    
    int xx = blockIdx.x * blockDim.x + threadIdx.x;
    int yy = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = yy * gridDim.x * blockDim.x + xx;
    int batch = blockIdx.y;
    idx = idx % (gridDim.x * blockDim.x * blockDim.y);
    if(idx < out_height * out_width * oc){
        
    
        int out_channel = idx / (out_height * out_width);
        int h = (idx % (out_height * out_width)) / out_width;
        int w = (idx % (out_height * out_width)) % out_width;
        output[w + h * out_width + out_channel * out_width * out_height 
                                    + batch * oc * out_width * out_height] = 0;
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

template <typename scalar_t>
__global__ void im2col(
    const scalar_t* __restrict__ im,
    scalar_t* __restrict__ col,
    int filter_height,
    int filter_width,
    int input_features,
    int out_height,
    int out_width,
    int strides_h,
    int strides_w,
    int in_height,
    int in_width,
    int batch)
{
    int xx = blockIdx.x * blockDim.x + threadIdx.x;
    int yy = blockIdx.y * blockDim.y + threadIdx.y;
    int index = yy * gridDim.x * blockDim.x + xx;

    int k = filter_height * filter_width * input_features;
    int num = out_height * out_width * batch;
    if(index < k * num){
        int h = index / num;
        int w = index % num;
        int n = w / (out_height * out_width);
        int out_idx = w % (out_height * out_width);
        int h_out = out_idx / out_width;
        int w_out = out_idx % out_width;
        int ic = h / (filter_height * filter_width);
        int hh_f = (h % (filter_height * filter_width)) / filter_width;
        int ww_f = (h % (filter_height * filter_width)) % filter_width;
        
        col[index] = im[ww_f + strides_w * w_out +
                        (hh_f + strides_h * h_out) * in_width +
                        ic * in_width * in_height +
                        n * in_width * in_height * input_features];
    }
}

template <typename scalar_t>
__global__ void col2im(
    const scalar_t* __restrict__ col,
    scalar_t* __restrict__ im,
    int filter_height,
    int filter_width,
    int input_features,
    int out_height,
    int out_width,
    int strides_h,
    int strides_w,
    int in_height,
    int in_width,
    int batch,
    int oc)
{
    int xx = blockIdx.x * blockDim.x + threadIdx.x;
    int yy = blockIdx.y * blockDim.y + threadIdx.y;
    int index = yy * gridDim.x * blockDim.x + xx;
    int num = out_height * out_width * batch;
    if(index < num * oc){
        int h = index / oc;
        int w = index % oc;
        int n = h / (out_height * out_width);
        int out_idx = h % (out_height * out_width);
        int h_out = out_idx / out_width;
        int w_out = out_idx % out_width;
        im[w_out + h_out * out_width + out_width * out_height * w + n * oc * out_width * out_height] = col[index];

    }
}


template <typename scalar_t>
__global__ void GEMM_CUDA_KERNEL(
    const scalar_t* __restrict__ col,
    const scalar_t* __restrict__ filter,
    const scalar_t* __restrict__ sign,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ result,
    int im_num,
    int filter_num,
    int k,
    int filter_height,
    int filter_width,
    int input_features)
{
    int xx = blockIdx.x * blockDim.x + threadIdx.x;
    int yy = blockIdx.y * blockDim.y + threadIdx.y;
    int index = yy * gridDim.x * blockDim.x + xx;
    if(index < filter_num * im_num){
        int h = index / filter_num;
        int w = index % filter_num; 
        for(int i = 0; i < k; i++){
            auto f = filter[i * filter_num + w];
            auto y = result[w + h * filter_num];
            auto x = col[h + i * im_num];
            
            auto s = sign[ i +
                          w * filter_height * filter_width * input_features];
            if((bool)s){
                if(f >= 0){
                    y -= (x << f);
                }
                else{
                    y -= (x >> (-f));
                }
            }
            else{
                if(f >= 0){
                    y += (x << f);
                }
                else{
                    y += (x >> (-f));
                }
            }

            result[w + h * filter_num] = y;
        }
        result[w + h * filter_num] += bias[w];

    }

}

void linear_shift_cuda(
    torch::Tensor& input,
    torch::Tensor& shift,
    torch::Tensor& sign,
    torch::Tensor& bias,
    torch::Tensor& output)
{
    

    const int block = (input.size(0) * shift.size(0) + 1024 -1) / 1024;
    const int threads = 1024;

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

      
}




void conv2d_shift_cuda(
    torch::Tensor& input,
    torch::Tensor& shift,
    torch::Tensor& sign,
    torch::Tensor& bias,
    torch::Tensor& output,
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
  
    int temp = (output.size(2) * output.size(3) * shift.size(0)  + 1024 - 1) / 1024;
    const dim3 block(temp, output.size(0));
    const dim3 threads(32,32);
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
            output.size(2),
            output.size(3),
            strides_h,
            strides_w,
            shift.size(0),
            input.size(3),
            input.size(2));
    })); 
    
}


void GEMM_CUDA(
    torch::Tensor& input,
    torch::Tensor& shift,
    torch::Tensor& sign,
    torch::Tensor& bias,
    torch::Tensor& output,
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
    
    int k  = shift.size(2)*shift.size(3)*shift.size(1);
    int num_p = output.size(2)*output.size(3)*output.size(0);
    auto options =  torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA, 0);
    auto col = torch::zeros({k, num_p},options);
    
    int tmp = (k * num_p + 1024 -1) / 1024;
    int tmp1 = (tmp + 65535 -1) / 65535;
    tmp  = (tmp > 65535) ? 65535: tmp;  
    const dim3 blk(tmp,tmp1);
    AT_DISPATCH_INTEGRAL_TYPES(input.type(), "im2col cuda", ([&] {
        im2col<scalar_t><<<blk, 1024>>>(
            input.data<scalar_t>(), 
            col.data<scalar_t>(),
            shift.size(2),
            shift.size(3),
            input.size(1),
            output.size(2),
            output.size(3),
            strides_h,
            strides_w,
            input.size(2),
            input.size(3),
            input.size(0));
    }));
    int filter_p = 1 * 1 * shift.size(0);
    auto filter = torch::zeros({k, filter_p},options);
    tmp = (k * filter_p + 1024 -1) / 1024;
    tmp1 = (tmp + 65535 -1) / 65535;
    tmp  = (tmp > 65535) ? 65535: tmp;
    const dim3 block(tmp,tmp1);
    AT_DISPATCH_INTEGRAL_TYPES(shift.type(), "im2col cuda", ([&] {
        im2col<scalar_t><<<block, 1024>>>(
            shift.data<scalar_t>(), 
            filter.data<scalar_t>(),
            shift.size(2),
            shift.size(3),
            shift.size(1),
            1,
            1,
            strides_h,
            strides_w,
            shift.size(2),
            shift.size(3),
            shift.size(0));
    }));

    tmp = (num_p * filter_p + 1024 -1) / 1024;
    tmp1 = (tmp + 65535 -1) / 65535;
    tmp  = (tmp > 65535) ? 65535: tmp;
    const dim3 block1(tmp,tmp1);
    auto result = torch::zeros({num_p, filter_p},options);
    AT_DISPATCH_INTEGRAL_TYPES(shift.type(), "GEMM_CUDA_KERNEL", ([&] {
        GEMM_CUDA_KERNEL<scalar_t><<<block1, 1024>>>(
            col.data<scalar_t>(), 
            filter.data<scalar_t>(),
            sign.data<scalar_t>(),
            bias.data<scalar_t>(),
            result.data<scalar_t>(),
            num_p,
            filter_p,
            k,
            shift.size(2),
            shift.size(3),
            shift.size(1));
    }));
    AT_DISPATCH_INTEGRAL_TYPES(result.type(), "col2im cuda", ([&] {
        col2im<scalar_t><<<block1, 1024>>>(
            result.data<scalar_t>(), 
            output.data<scalar_t>(),
            shift.size(2),
            shift.size(3),
            input.size(1),
            output.size(2),
            output.size(3),
            strides_h,
            strides_w,
            input.size(2),
            input.size(3),
            input.size(0),
            shift.size(0));
    }));
  
}