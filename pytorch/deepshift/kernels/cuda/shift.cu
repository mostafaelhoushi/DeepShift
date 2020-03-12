
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ctime>
#include <vector>
#define BLOCK_SIZE 16
#define MAX_BITS 32
#define MAX_THREADS 1024
#define MAX_BLOCKS 65535
#define ZERO_BASE 1
#define NON_ZERO_BASE 0
#define BIT_1 1
#define BIT_2 2
#define BIT_3 3
#define BIT_4 4
#define BIT_5 5
#define BIT_6 6
#define BIT_7 7
#define NUM_4 4
#define NUM_5 5
#define NUM_6 6
#define NUM_8 8
#define NUM_10 10
#define NUM_16 16

__device__ int COMPRESS(const int* __restrict__ shift, const int* __restrict__ sign, int length, int base, int bits)
{
    int value = 0;
    int s = 0;
    for (int i = 0; i < length; i++) {
        value  = (value) | ((shift[i] - base) << s);
        s = s + bits;
        value  = (value) | ((sign[i] > 0 ? 1 : 0) << s);
        s = s + 1;
    }
    return value;
}

__global__ void COMPRESS_SIGN_SHIFT_GPU_KERNEL( int* __restrict__ shift,  int* __restrict__ sign, int* __restrict__ weight, int oc, int in_c, int height, int width, int num,int base, int bits, int compressed_row_length, int row_length)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x; 
    if(index < compressed_row_length) {
        int* shift_sub = &shift[oc * in_c * height * width];
        int* sign_sub = &sign[oc * in_c * height * width];
        int* weight_sub = &weight[oc * compressed_row_length];
        int length = num;
        if( (index + 1) * num >= in_c * height * width) {
            length = in_c * height * width - index * num;
        }
        weight_sub[index] = COMPRESS(&shift_sub[index * num], &sign_sub[index * num], length, base, bits); 
        
    }
    __syncthreads();
}

template <int num, int bits, char mask_shift, char mask_sign, bool zero_base>
__global__ void DEEP_SHIFT_GEMM_GPU_KERNEL(
    const int* __restrict__ input,
    const int* __restrict__ shift,
    const int* __restrict__ bias,
    int* __restrict__ output,
    const int n,
    const int m,
    const int k, 
    const int base,
    const int max,
    const int row_length) 
{
    const int row = threadIdx.y;
    const int col = threadIdx.x;
    for(int blockRow = blockIdx.y;blockDim.y * blockRow < m; blockRow = blockRow + gridDim.y){
        for(int blockCol = blockIdx.x;blockDim.x * blockCol < k; blockCol = blockCol + gridDim.x){
            
            const int compressed_row = row_length / num;
            int* Csub = &output[BLOCK_SIZE * k * blockRow + BLOCK_SIZE * blockCol];
            __shared__ int As[BLOCK_SIZE *BLOCK_SIZE * num];
            __shared__ int Bs[BLOCK_SIZE *BLOCK_SIZE];
            int Cvalue = 0;
            for (int i = 0; i < max; ++i) {
                const int* Asub = &input[BLOCK_SIZE * blockRow * n + BLOCK_SIZE * i * num ];
                const int* Bsub = &shift[(BLOCK_SIZE * blockCol * row_length + BLOCK_SIZE * i * num) / num];
                #pragma unroll
                for( int d = 0; d < num; d++) {
                        As[row * BLOCK_SIZE * num + col * num + d] = Asub[row * n + col * num + d];
                }
                Bs[row * BLOCK_SIZE + col] = Bsub[row * compressed_row + col];
                
                __syncthreads();
               
                #pragma unroll
                for (int j = 0; j < BLOCK_SIZE ; ++j){
                    if(col + blockCol* BLOCK_SIZE< k 
                        && row + blockRow* BLOCK_SIZE< m ){
                        int whole = Bs[col * BLOCK_SIZE + j];
                        #pragma unroll
                        for(int d = 0; d < num; d++) {
                            if(i * BLOCK_SIZE * num + j * num + d < n){
                                int get_sign = int(whole & mask_sign);
                                get_sign = get_sign == 0 ? -1 : 1;
                                int get_shift = int(whole & mask_shift);
                                whole = int(whole >> (bits + 1));
                                if(zero_base){
                                    Cvalue += get_sign * (As[row * BLOCK_SIZE * num+ j * num + d] >> (get_shift));
                                }
                                else{
                                    Cvalue += get_sign * ((As[row * BLOCK_SIZE * num+ j * num + d] >> (get_shift))<<(-base));
                                }
                            }
                        }
                    }
                } 
                __syncthreads();
            }
            if(col + blockCol* BLOCK_SIZE< k && row + blockRow* BLOCK_SIZE< m) Csub[row*k+col] = Cvalue + bias[col + blockCol* BLOCK_SIZE];
            __syncthreads();
        }
        
    }
}

__global__ void IM2COL(
    const int total,
    const int* __restrict__ im,
    int* __restrict__ col,
    const int filter_height,
    const int filter_width,
    const int input_features,
    const int out_height,
    const int out_width,
    const int strides_h,
    const int strides_w,
    const int in_height,
    const int in_width,
    const int k)
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
    const int* __restrict__ col,
    int* __restrict__ im,
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

void DEEP_SHIFT_LINEAR_GPU(
    torch::Tensor input,
    torch::Tensor shift,
    torch::Tensor bias,
    torch::Tensor output,
    int base, int bits, int out_features)
{
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    int a1=out_features/ BLOCK_SIZE + 1;
    if(a1> MAX_BLOCKS){
        a1 = MAX_BLOCKS;
    }
    int a2=input.size(0) / BLOCK_SIZE + 1;
    if(a2> MAX_BLOCKS) {
        a2= MAX_BLOCKS;
    }
    dim3 gridDim( a1, a2);
    int num = int(MAX_BITS / (bits + 1));
    int comm = (input.size(1) + num - 1) / num;
    int max =(comm + BLOCK_SIZE - 1) / BLOCK_SIZE;
    if(bits == 1) {
        if(base == 0){
            AT_DISPATCH_INTEGRAL_TYPES(input.type(), "linear shift kernel", ([&] {
                DEEP_SHIFT_GEMM_GPU_KERNEL<NUM_16, BIT_1, 0x01,0x02, ZERO_BASE><<<gridDim, blockDim>>>(
                    input.data<int>(),
                    shift.data<int>(),
                    bias.data<int>(),
                    output.data<int>(),
                    input.size(1),
                    input.size(0),
                    out_features, base,max, comm * num);
              }));
        }
        else {
            AT_DISPATCH_INTEGRAL_TYPES(input.type(), "linear shift kernel", ([&] {
                DEEP_SHIFT_GEMM_GPU_KERNEL<NUM_16, BIT_1, 0x01,0x02, NON_ZERO_BASE><<<gridDim, blockDim>>>(
                    input.data<int>(),
                    shift.data<int>(),
                    bias.data<int>(),
                    output.data<int>(),
                    input.size(1),
                    input.size(0),
                    out_features, base,max, comm * num);
              }));
        }

    }
    else if(bits == 2) {
        if(base == 0){
            AT_DISPATCH_INTEGRAL_TYPES(input.type(), "linear shift kernel", ([&] {
                DEEP_SHIFT_GEMM_GPU_KERNEL<NUM_10, BIT_2, 0x03,0x04, ZERO_BASE><<<gridDim, blockDim>>>(
                    input.data<int>(),
                    shift.data<int>(),
                    bias.data<int>(),
                    output.data<int>(),
                    input.size(1),
                    input.size(0),
                    out_features, base,max, comm * num);
              }));
        }
        else {
            AT_DISPATCH_INTEGRAL_TYPES(input.type(), "linear shift kernel", ([&] {
                DEEP_SHIFT_GEMM_GPU_KERNEL<NUM_10, BIT_2, 0x03,0x04, NON_ZERO_BASE><<<gridDim, blockDim>>>(
                    input.data<int>(),
                    shift.data<int>(),
                    bias.data<int>(),
                    output.data<int>(),
                    input.size(1),
                    input.size(0),
                    out_features, base,max, comm * num);
              }));
        }

    }
    else if(bits == 3) {
        if(base == 0){
            AT_DISPATCH_INTEGRAL_TYPES(input.type(), "linear shift kernel", ([&] {
                DEEP_SHIFT_GEMM_GPU_KERNEL<NUM_8, BIT_3, 0x07,0x08, ZERO_BASE><<<gridDim, blockDim>>>(
                    input.data<int>(),
                    shift.data<int>(),
                    bias.data<int>(),
                    output.data<int>(),
                    input.size(1),
                    input.size(0),
                    out_features, base,max, comm * num);
              }));
        }
        else {
            AT_DISPATCH_INTEGRAL_TYPES(input.type(), "linear shift kernel", ([&] {
                DEEP_SHIFT_GEMM_GPU_KERNEL<NUM_8, BIT_3, 0x07,0x08, NON_ZERO_BASE><<<gridDim, blockDim>>>(
                    input.data<int>(),
                    shift.data<int>(),
                    bias.data<int>(),
                    output.data<int>(),
                    input.size(1),
                    input.size(0),
                    out_features, base,max, comm * num);
              }));
        }
        
    }
    else if(bits == 4){
        if(base == 0){
            AT_DISPATCH_INTEGRAL_TYPES(input.type(), "linear shift kernel", ([&] {
                DEEP_SHIFT_GEMM_GPU_KERNEL<NUM_6, BIT_4, 0x0f,0x10, ZERO_BASE><<<gridDim, blockDim>>>(
                    input.data<int>(),
                    shift.data<int>(),
                    bias.data<int>(),
                    output.data<int>(),
                    input.size(1),
                    input.size(0),
                    out_features, base,max, comm * num);
              }));
        }
        else {
            AT_DISPATCH_INTEGRAL_TYPES(input.type(), "linear shift kernel", ([&] {
                DEEP_SHIFT_GEMM_GPU_KERNEL<NUM_6, BIT_4, 0x0f,0x10, NON_ZERO_BASE><<<gridDim, blockDim>>>(
                    input.data<int>(),
                    shift.data<int>(),
                    bias.data<int>(),
                    output.data<int>(),
                    input.size(1),
                    input.size(0),
                    out_features, base,max, comm * num);
              }));
        }
        
    }
    else if(bits == 5){
        if(base == 0){
            AT_DISPATCH_INTEGRAL_TYPES(input.type(), "linear shift kernel", ([&] {
                DEEP_SHIFT_GEMM_GPU_KERNEL<NUM_5, BIT_5, 0x1f,0x20,ZERO_BASE><<<gridDim, blockDim >>>(
                    input.data<int>(),
                    shift.data<int>(),
                    bias.data<int>(),
                    output.data<int>(),
                    input.size(1),
                    input.size(0),
                    out_features,base, max, comm * num);
              }));
        }
        else {
            AT_DISPATCH_INTEGRAL_TYPES(input.type(), "linear shift kernel", ([&] {
                DEEP_SHIFT_GEMM_GPU_KERNEL<NUM_5, BIT_5, 0x1f,0x20,NON_ZERO_BASE><<<gridDim, blockDim >>>(
                    input.data<int>(),
                    shift.data<int>(),
                    bias.data<int>(),
                    output.data<int>(),
                    input.size(1),
                    input.size(0),
                    out_features,base, max, comm * num);
              }));
        }
        
    }
    else if(bits == 6){
        if(base == 0){
            AT_DISPATCH_INTEGRAL_TYPES(input.type(), "linear shift kernel", ([&] {
                DEEP_SHIFT_GEMM_GPU_KERNEL<NUM_4, BIT_6, 0x3f,0x40, ZERO_BASE><<<gridDim, blockDim >>>(
                    input.data<int>(),
                    shift.data<int>(),
                    bias.data<int>(),
                    output.data<int>(),
                    input.size(1),
                    input.size(0),
                    out_features, base, max, comm * num);
              }));
        }
        else{
            AT_DISPATCH_INTEGRAL_TYPES(input.type(), "linear shift kernel", ([&] {
                DEEP_SHIFT_GEMM_GPU_KERNEL<NUM_4, BIT_6, 0x3f,0x40, NON_ZERO_BASE><<<gridDim, blockDim >>>(
                    input.data<int>(),
                    shift.data<int>(),
                    bias.data<int>(),
                    output.data<int>(),
                    input.size(1),
                    input.size(0),
                    out_features, base,max, comm * num);
              }));
        }
        
    }
    else if(bits == 7){
        if(base == 0){
            AT_DISPATCH_INTEGRAL_TYPES(input.type(), "linear shift kernel", ([&] {
                DEEP_SHIFT_GEMM_GPU_KERNEL<NUM_4, BIT_7, 0x7f,0x80, ZERO_BASE><<<gridDim, blockDim>>>(
                    input.data<int>(),
                    shift.data<int>(),
                    bias.data<int>(),
                    output.data<int>(),
                    input.size(1),
                    input.size(0),
                    out_features, base,max, comm * num);
              }));
        }
        else {
            AT_DISPATCH_INTEGRAL_TYPES(input.type(), "linear shift kernel", ([&] {
                DEEP_SHIFT_GEMM_GPU_KERNEL<NUM_4, BIT_7, 0x7f,0x80, NON_ZERO_BASE><<<gridDim, blockDim>>>(
                    input.data<int>(),
                    shift.data<int>(),
                    bias.data<int>(),
                    output.data<int>(),
                    input.size(1),
                    input.size(0),
                    out_features, base,max, comm * num);
              }));
        }
        
    }
    else{
        std::cout<<"ERROR: unhandled case\n";
    }
      
}
void COMPRESS_SIGN_SHIFT_GPU(torch::Tensor shift, torch::Tensor sign, torch::Tensor weight, int base, int bits, int out_c, int in_c, int height, int width, int row_length, int num)
{
    int threads = MAX_THREADS;
    int compressed_row_length = row_length;
    dim3 block ( (compressed_row_length + threads - 1) / threads);
    for(int i = 0; i < out_c; i++) {
        COMPRESS_SIGN_SHIFT_GPU_KERNEL<<<block, threads>>>(shift.data<int>(), sign.data<int>(),weight.data<int>(),
                                                    i, in_c, height, width, num, base, bits,compressed_row_length, row_length);
    }
}

void DEEP_SHIFT_CONV_GPU(torch::Tensor data_im,
                torch::Tensor shift,
                torch::Tensor bias,
                torch::Tensor output,
                torch::IntArrayRef strides, 
                torch::IntArrayRef padding, int filter_height, int filter_width, int base, int bits)
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
    int k  = filter_height * filter_width * data_im.size(1);
    int num_patch = output.size(0) * output.size(2) * output.size(3);

    int* data_col;
    cudaMalloc(&data_col, num_patch * k * sizeof(int));

    int threads = MAX_THREADS;
    int tmp = (k * num_patch + threads -1) / threads;
    tmp  = (tmp > MAX_BLOCKS) ? MAX_BLOCKS: tmp;  
    const dim3 blk(tmp);
    AT_DISPATCH_INTEGRAL_TYPES(data_im.type(), "IM2COL cuda", ([&] {
        IM2COL<<<blk, threads>>>(
            k * num_patch,
            data_im.data<int>(), 
            data_col,
            filter_height,
            filter_width,
            data_im.size(1),
            output.size(2),
            output.size(3),
            strides_h,
            strides_w,
            data_im.size(2),
            data_im.size(3),
            k);
    }));

    int filter_patch = output.size(1);
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    
    int a1=filter_patch/ BLOCK_SIZE + 1;
    if(a1> MAX_BLOCKS){
        a1 = MAX_BLOCKS;  
    }
    int a2=num_patch  / BLOCK_SIZE + 1;
    if(a2> MAX_BLOCKS) {
        a2 = MAX_BLOCKS;
    }
    dim3 gridDim( a1, a2);

    int *out_col;
    int num = int(MAX_BITS / (bits +1 ));
    int comm = (k + num -1 ) / num;
    int max =(comm + BLOCK_SIZE - 1) / BLOCK_SIZE;
    cudaMalloc(&out_col,  num_patch * filter_patch * sizeof(int));

    if(bits == 1) {
        if(base == 0){
            AT_DISPATCH_INTEGRAL_TYPES(data_im.type(), "DEEP_SHIFT_GEMM_GPU_KERNEL kernel", ([&] {
                DEEP_SHIFT_GEMM_GPU_KERNEL<NUM_16, BIT_1, 0x01,0x02, ZERO_BASE><<<gridDim, blockDim>>>(
                    data_col,
                    shift.data<int>(),
                    bias.data<int>(),
                    out_col,
                    k,
                    num_patch,
                    filter_patch, base,max,comm * num);
              }));
        }
        else {
            AT_DISPATCH_INTEGRAL_TYPES(data_im.type(), "DEEP_SHIFT_GEMM_GPU_KERNEL kernel", ([&] {
                DEEP_SHIFT_GEMM_GPU_KERNEL<NUM_16, BIT_1, 0x01,0x02, NON_ZERO_BASE><<<gridDim, blockDim>>>(
                    data_col,
                    shift.data<int>(),
                    bias.data<int>(),
                    out_col,
                    k,
                    num_patch,
                    filter_patch, base,max,comm * num);
              }));
        }

    }
    else if(bits == 2) {
        if(base == 0){
            AT_DISPATCH_INTEGRAL_TYPES(data_im.type(), "DEEP_SHIFT_GEMM_GPU_KERNEL kernel", ([&] {
                DEEP_SHIFT_GEMM_GPU_KERNEL<NUM_10, BIT_2, 0x03,0x04, ZERO_BASE><<<gridDim, blockDim>>>(
                    data_col,
                    shift.data<int>(),
                    bias.data<int>(),
                    out_col,
                    k,
                    num_patch,
                    filter_patch, base,max,comm * num);
              }));
        }
        else {
            AT_DISPATCH_INTEGRAL_TYPES(data_im.type(), "DEEP_SHIFT_GEMM_GPU_KERNEL kernel", ([&] {
                DEEP_SHIFT_GEMM_GPU_KERNEL<NUM_10, BIT_2, 0x03,0x04, NON_ZERO_BASE><<<gridDim, blockDim>>>(
                    data_col,
                    shift.data<int>(),
                    bias.data<int>(),
                    out_col,
                    k,
                    num_patch,
                    filter_patch, base,max,comm * num);
              }));
        }

    }
    else if(bits == 3) {
        if(base == 0){
            AT_DISPATCH_INTEGRAL_TYPES(data_im.type(), "DEEP_SHIFT_GEMM_GPU_KERNEL kernel", ([&] {  
                DEEP_SHIFT_GEMM_GPU_KERNEL<NUM_8, BIT_3, 0x07,0x08, ZERO_BASE><<<gridDim, blockDim>>>(
                    data_col,
                    shift.data<int>(),
                    bias.data<int>(),
                    out_col,
                    k,
                    num_patch,
                    filter_patch, base,max,comm * num);
              }));
        }
        else {
            AT_DISPATCH_INTEGRAL_TYPES(data_im.type(), "DEEP_SHIFT_GEMM_GPU_KERNEL kernel", ([&] {  
                DEEP_SHIFT_GEMM_GPU_KERNEL<NUM_8, BIT_3, 0x07,0x08, NON_ZERO_BASE><<<gridDim, blockDim>>>(
                    data_col,
                    shift.data<int>(),
                    bias.data<int>(),
                    out_col,
                    k,
                    num_patch,
                    filter_patch, base,max,comm * num);
              }));
        }
        
    }
    else if(bits == 4) {
        if(base == 0){
            AT_DISPATCH_INTEGRAL_TYPES(data_im.type(), "DEEP_SHIFT_GEMM_GPU_KERNEL kernel", ([&] {  
                DEEP_SHIFT_GEMM_GPU_KERNEL<NUM_6, BIT_4, 0x0f,0x10, ZERO_BASE><<<gridDim, blockDim>>>(
                    data_col,
                    shift.data<int>(),
                    bias.data<int>(),
                    out_col,
                    k,
                    num_patch,
                    filter_patch, base,max,comm * num);
              }));
        }
        else {
            AT_DISPATCH_INTEGRAL_TYPES(data_im.type(), "DEEP_SHIFT_GEMM_GPU_KERNEL kernel", ([&] {  
                DEEP_SHIFT_GEMM_GPU_KERNEL<NUM_6, BIT_4, 0x0f,0x10, NON_ZERO_BASE><<<gridDim, blockDim>>>(
                    data_col,
                    shift.data<int>(),
                    bias.data<int>(),
                    out_col,
                    k,
                    num_patch,
                    filter_patch, base,max,comm * num);
              }));
        }
        
    }
    else if(bits == 5){
        if(base == 0){
            AT_DISPATCH_INTEGRAL_TYPES(data_im.type(), "DEEP_SHIFT_GEMM_GPU_KERNEL kernel", ([&] {  
                DEEP_SHIFT_GEMM_GPU_KERNEL<NUM_5, BIT_5, 0x1f,0x20, ZERO_BASE><<<gridDim, blockDim>>>(
                    data_col,
                    shift.data<int>(),
                    bias.data<int>(),
                    out_col,
                    k,
                    num_patch,
                    filter_patch, base,max,comm * num);
              }));
        }
        else {
            AT_DISPATCH_INTEGRAL_TYPES(data_im.type(), "DEEP_SHIFT_GEMM_GPU_KERNEL kernel", ([&] {  
                DEEP_SHIFT_GEMM_GPU_KERNEL<NUM_5, BIT_5, 0x1f,0x20, NON_ZERO_BASE><<<gridDim, blockDim>>>(
                    data_col,
                    shift.data<int>(),
                    bias.data<int>(),
                    out_col,
                    k,
                    num_patch,
                    filter_patch, base,max,comm * num);
              }));
        }
        
    }
    else if(bits == 6){
        if(base == 0){
            AT_DISPATCH_INTEGRAL_TYPES(data_im.type(), "DEEP_SHIFT_GEMM_GPU_KERNEL kernel", ([&] {  
                DEEP_SHIFT_GEMM_GPU_KERNEL<NUM_4, BIT_6, 0x3f,0x40, ZERO_BASE><<<gridDim, blockDim>>>(
                    data_col,
                    shift.data<int>(),
                    bias.data<int>(),
                    out_col,
                    k,
                    num_patch,
                    filter_patch, base,max,comm * num);
              }));
        }
        else{
            AT_DISPATCH_INTEGRAL_TYPES(data_im.type(), "DEEP_SHIFT_GEMM_GPU_KERNEL kernel", ([&] {  
                DEEP_SHIFT_GEMM_GPU_KERNEL<NUM_4, BIT_6, 0x3f,0x40, NON_ZERO_BASE><<<gridDim, blockDim>>>(
                    data_col,
                    shift.data<int>(),
                    bias.data<int>(),
                    out_col,
                    k,
                    num_patch,
                    filter_patch, base,max,comm * num);
              }));
        }
        
    }
    else if(bits == 7){
        if(base == 0){
            AT_DISPATCH_INTEGRAL_TYPES(data_im.type(), "DEEP_SHIFT_GEMM_GPU_KERNEL kernel", ([&] {  
                DEEP_SHIFT_GEMM_GPU_KERNEL<NUM_4, BIT_7, 0x7f,0x80, ZERO_BASE><<<gridDim, blockDim>>>(
                    data_col,
                    shift.data<int>(),
                    bias.data<int>(),
                    out_col,
                    k,
                    num_patch,
                    filter_patch, base,max,comm * num);
              }));
        }
        else {
            AT_DISPATCH_INTEGRAL_TYPES(data_im.type(), "DEEP_SHIFT_GEMM_GPU_KERNEL kernel", ([&] {  
                DEEP_SHIFT_GEMM_GPU_KERNEL<NUM_4, BIT_7, 0x7f,0x80, NON_ZERO_BASE><<<gridDim, blockDim>>>(
                    data_col,
                    shift.data<int>(),
                    bias.data<int>(),
                    out_col,
                    k,
                    num_patch,
                    filter_patch, base,max,comm * num);
              }));  
        }
        
    }
    else{
        std::cout<<"ERROR: unhandled case\n";
    }

    tmp = (num_patch * output.size(1) + threads -1) / threads;
    tmp  = (tmp > MAX_BLOCKS) ? MAX_BLOCKS: tmp;
    const dim3 block1(tmp);
    AT_DISPATCH_INTEGRAL_TYPES(data_im.type(), "COL2IM cuda", ([&] {
        COL2IM<<<block1, threads>>>(
            num_patch * output.size(1),
            out_col, 
            output.data<int>(),
            output.size(2),
            output.size(3),
            output.size(1));
    }));
    cudaFree(data_col);
    cudaFree(out_col);
}



