
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ctime>
#include <vector>
#define BLOCK_SIZE 16
#define MAX_THREADS 1024
#define MAX_BLOCKS 65535
__global__ void IM2COL(
    const int total,
    const float* __restrict__ im,
    float* __restrict__ col,
    const int filter_height,
    const int filter_width,
    const int input_features,
    const int out_height,
    const int out_width,
    const int strides_h,
    const int strides_w,
    const int in_height,
    const int in_width,
    const int k, const int num)
{
    for(int index = blockIdx.x * blockDim.x + threadIdx.x; index < total; index = index + gridDim.x * blockDim.x) {
        const int h = index / k;
        const int w = index % k;
        const int n = h / (out_height * out_width);
        const int out_idx = h % (out_height * out_width);
        const int h_out = out_idx / out_width;
        const int w_out = out_idx % out_width;
        const int ic = w / (filter_height * filter_width);
        const int hh_f = (w % (filter_height * filter_width)) / filter_width;
        const int ww_f = (w % (filter_height * filter_width)) % filter_width;
        
        col[index] = im[ww_f + strides_w * w_out +
                        (hh_f + strides_h * h_out) * in_width +
                        ic * in_width * in_height +
                        n * in_width * in_height * input_features];
    }     
}


__global__ void COL2IM(
    const int total, 
    const float* __restrict__ col,
    float* __restrict__ im,
    const int out_height,
    const int out_width,
    const int oc)
{
    for(int index = blockIdx.x * blockDim.x + threadIdx.x; index < total; index = index + gridDim.x * blockDim.x){
        const int h = index / oc;
        const int w = index % oc;
        const int n = h / (out_height * out_width);
        const int out_idx = h % (out_height * out_width);
        const int h_out = out_idx / out_width;
        const int w_out = out_idx % out_width;
        im[w_out + h_out * out_width + out_width * out_height * w + n * oc * out_width * out_height] = col[index];
    }
}

__global__ void GEMM(
    const float* __restrict__ input,
    const float* __restrict__ shift,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int n,
    const int m,
    const int k, 
    const int max) 
{

    const int row = threadIdx.y;
    const int col = threadIdx.x;
    for(int blockRow = blockIdx.y;blockDim.y * blockRow < m; blockRow = blockRow + gridDim.y){
        for(int blockCol = blockIdx.x;blockDim.x * blockCol < k; blockCol = blockCol + gridDim.x){
            float* Csub = &output[BLOCK_SIZE * k * blockRow + BLOCK_SIZE * blockCol];
            __shared__ float As[BLOCK_SIZE*BLOCK_SIZE];
            __shared__ float Bs[BLOCK_SIZE*BLOCK_SIZE];
            float Cvalue = 0;
            for (int i = 0; i < max; ++i) {
                const float* Asub = &input[BLOCK_SIZE * blockRow * n + BLOCK_SIZE * i ];
                const int original_index = BLOCK_SIZE * blockCol * n + BLOCK_SIZE * i  + row * n + col;
                As[row * BLOCK_SIZE + col] = Asub[row*n+col];
                Bs[row * BLOCK_SIZE + col] = shift[(original_index)];
                __syncthreads();
                
                #pragma unroll
                for (int j = 0; j < BLOCK_SIZE ; ++j){
                    if(col + blockCol* BLOCK_SIZE< k 
                        && row + blockRow* BLOCK_SIZE< m
                        && i * BLOCK_SIZE  + j < n){
                            Cvalue += (As[row * BLOCK_SIZE + j] * Bs[col * BLOCK_SIZE + j]);
                    }
                } 
                __syncthreads();
            }
            if(col + blockCol* BLOCK_SIZE< k && row + blockRow* BLOCK_SIZE< m) Csub[row*k+col] = Cvalue + bias[col + blockCol* BLOCK_SIZE];
        }
        __syncthreads();
    }
}


void UNOPTIMIZED_LINEAR_GPU(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor output)
{
    
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    int a1=weight.size(0)/ BLOCK_SIZE + 1;
    if(a1> MAX_BLOCKS){
        a1 = MAX_BLOCKS;
    }
    int a2=input.size(0) / BLOCK_SIZE + 1;
    if(a2> MAX_BLOCKS) {
        a2= MAX_BLOCKS;
    }
    dim3 gridDim( a1, a2);
    int max =(input.size(1) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    AT_DISPATCH_ALL_TYPES(input.type(), "linear unoptimized kernel", ([&] {
        GEMM<<<gridDim, blockDim >>>(
            input.data<float>(),
            weight.data<float>(),
            bias.data<float>(),
            output.data<float>(),
            input.size(1),
            input.size(0),
            weight.size(0), max);
      }));
}

void UNOPTIMIZED_CONV_GPU(
    torch::Tensor data_im,
    torch::Tensor shift,
    torch::Tensor bias,
    torch::Tensor output,
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
    int k  = shift.size(2) * shift.size(3) * data_im.size(1);
    int num_p = output.size(0) * output.size(2) * output.size(3);

    float* data_col;
    cudaMalloc(&data_col, num_p * k * sizeof(float));

    int threads = MAX_THREADS;
    int tmp = (k * num_p + threads -1) / threads;
    tmp  = (tmp > MAX_BLOCKS) ? MAX_BLOCKS: tmp;  
    const dim3 blk(tmp);
    AT_DISPATCH_ALL_TYPES(data_im.type(), "IM2COL cuda", ([&] {
        IM2COL<<<blk, threads>>>(
        k * num_p,
        data_im.data<float>(), 
        data_col,
        shift.size(2),
        shift.size(3),
        data_im.size(1),
        output.size(2),
        output.size(3),
        strides_h,
        strides_w,
        data_im.size(2),
        data_im.size(3),
        k, num_p);
        }));
    int filter_p = output.size(1);
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    int a1=filter_p/ BLOCK_SIZE + 1;
    if(a1> MAX_BLOCKS){
        a1 = MAX_BLOCKS;   
    }
    int a2=num_p  / BLOCK_SIZE + 1;
    if(a2> MAX_BLOCKS) {
        a2 = MAX_BLOCKS;
    }
    dim3 gridDim( a1, a2);

    float *out_col;
    int max =(k + BLOCK_SIZE - 1) / BLOCK_SIZE;
    cudaMalloc(&out_col,  num_p * filter_p * sizeof(float));
    AT_DISPATCH_ALL_TYPES(data_im.type(), "GEMM unoptimized kernel", ([&] {  
        GEMM<<<gridDim, blockDim>>>(
        data_col,
        shift.data<float>(),
        bias.data<float>(),
        out_col,
        k,
        num_p,
        filter_p, max);
        }));
    tmp = (num_p * output.size(1) + threads -1) / threads;
    tmp  = (tmp > MAX_BLOCKS) ? MAX_BLOCKS: tmp;
    const dim3 block1(tmp);
    AT_DISPATCH_ALL_TYPES(data_im.type(), "COL2IM cuda", ([&] {
        COL2IM<<<block1, threads>>>(
            num_p * output.size(1),
            out_col, 
            output.data<float>(),
            output.size(2),
            output.size(3),
            output.size(1));
            }));
    cudaFree(data_col);
    cudaFree(out_col);
}