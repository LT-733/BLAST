#ifndef level2
#define level2
#include <cuda_runtime.h>
#include <math.h>
#include <thrust/complex.h>

constexpr int BLOCK_SIZE = 256;
constexpr int WARP_SIZE = 32;
constexpr int WARP_PER_BLOCK = 8;

__global__ void gemv_kernel(unsigned int m, unsigned int n, const half* __restrict__ A, const half* __restrict__ x, half* __restrict__ y, half alpha, half beta){
    auto ceiling_division = [](unsigned int x, unsigned int y){
        return (x % y != 0) ? (x / y + 1) : x/y;
    };
    unsigned int warp_id = threadIdx.x / WARP_SIZE;
    unsigned int warp_row = blockIdx.x * WARP_PER_BLOCK + warp_id;
    if (warp_row < m){
        float temp = 0.0f;
        // each thread needs to 
        unsigned int ith_turn_to_n = ceiling_division(n, WARP_SIZE);
        unsigned int leftover_idx = threadIdx.x % WARP_SIZE;
        #pragma unroll
        for(unsigned int i = 0; i < ith_turn_to_n; ++i){
            unsigned int xidx = i * WARP_SIZE + leftover_idx;
            unsigned int aidx = n * warp_row + (i * WARP_SIZE + leftover_idx);
            if(xidx < n) temp += __half2float(A[aidx]) * __half2float(x[xidx]);
        }
        const unsigned int mask = 0xffffffff;
        #pragma unroll
        for(unsigned int i = WARP_SIZE/2; i >= 1; i /=2){
            temp += __shfl_down_sync(mask, temp, i);
        }
        if(leftover_idx == 0) y[warp_row] = __float2half(__half2float(alpha) * temp + __half2float(beta) * __half2float(y[warp_row]));
    }
}

__global__ void trmv(unsigned int n, const char uplo, const half* __restrict__ A, const half* __restrict__ x, half* __restrict__ res){
    // auto ceiling_division = [](unsigned int x, unsigned int y){
    //     return (x % y != 0) ? (x / y + 1) : x/y;
    // };
    int which_row_am_i = blockDim.x * blockIdx.x + threadIdx.x;
    if(which_row_am_i < n){
        float tmp = 0.0f;
        if(uplo == 'U'){
            #pragma unroll
            for(int i = which_row_am_i; i < n; ++i){
                tmp += __half2float(A[which_row_am_i * n + i]) * __half2float(x[i]);
            }
        }else if(uplo == 'L'){
            #pragma unroll
            for(int i = 0; i <= which_row_am_i; ++i){
                tmp += __half2float(A[which_row_am_i * n + i]) * __half2float(x[i]);
            }
        }
        res[which_row_am_i] = __float2half(tmp);
    }
}

__global__ void trmv_optimized(unsigned int n, const char uplo, const half* __restrict__ A, const half* __restrict__ x, half* __restrict__ res){
        auto ceiling_division = [](unsigned int x, unsigned int y){
        return (x % y != 0) ? (x / y + 1) : x/y;
    };
    unsigned int warp_id = threadIdx.x / WARP_SIZE;
    unsigned int warp_row = blockIdx.x * WARP_PER_BLOCK + warp_id;
    if (warp_row < n){
        float temp = 0.0f;
        // each thread needs to 
        unsigned int ith_turn_to_n = (uplo=='U') ? ceiling_division((n - warp_row), WARP_SIZE) : ceiling_division((warp_row + 1), WARP_SIZE);
        unsigned int leftover_idx = threadIdx.x % WARP_SIZE;
        #pragma unroll
        for(unsigned int i = 0; i < ith_turn_to_n; ++i){
            unsigned int xidx = (uplo=='U') ? warp_row + i * WARP_SIZE + leftover_idx : i * WARP_SIZE + leftover_idx;
            unsigned int aidx = n * warp_row + xidx;
            if(xidx < n) temp += __half2float(A[aidx]) * __half2float(x[xidx]);
        }
        const unsigned int mask = 0xffffffff;
        #pragma unroll
        for(unsigned int i = WARP_SIZE/2; i >= 1; i /=2){
            temp += __shfl_xor_sync(mask, temp, i);
        }
        if(leftover_idx == 0) res[warp_row] = __float2half(temp);
    }
}

__global__ void ger(unsigned int m, unsigned int n, half* __restrict__ A, const half* __restrict__ x, const half* __restrict__ y, half alpha){
    int idx_inA = blockDim.x * blockIdx.x + threadIdx.x;
    int row_inA = idx_inA / n;
    int col_inA = idx_inA % n;
    if(idx_inA < m * n) A[idx_inA] += __float2half(__half2float(alpha) * __half2float(x[row_inA]) * __half2float(y[col_inA]));
}

__global__ void syr(unsigned int n, const char uplo, half* __restrict__ A, const half* __restrict__ x, half alpha) {
    int idx_inA = blockDim.x * blockIdx.x + threadIdx.x;
    int row_inA = idx_inA / n;
    int col_inA = idx_inA % n;
    if(idx_inA < n * n && uplo == 'L' && col_inA <= row_inA) A[idx_inA] += __float2half(__half2float(alpha) * __half2float(x[row_inA]) * __half2float(x[col_inA]));
    if(idx_inA < n * n && uplo == 'U' && col_inA >=row_inA) A[idx_inA] += __float2half(__half2float(alpha) * __half2float(x[row_inA]) * __half2float(x[col_inA]));
}

__global__ void syr2(unsigned int n, const char uplo, half* __restrict__ A, const half* __restrict__ x, const half* __restrict__ y, half alpha) {
    int idx_inA = blockDim.x * blockIdx.x + threadIdx.x;
    int row_inA = idx_inA / n;
    int col_inA = idx_inA % n;
    if(idx_inA < n * n && uplo == 'L' && col_inA <= row_inA) A[idx_inA] += __float2half(__half2float(alpha) * __half2float(x[row_inA]) * __half2float(y[col_inA]) + __half2float(alpha) * __half2float(y[row_inA]) * __half2float(x[col_inA]));
    if(idx_inA < n * n && uplo == 'U' && col_inA >=row_inA) A[idx_inA] += __float2half(__half2float(alpha) * __half2float(x[row_inA]) * __half2float(y[col_inA]) + __half2float(alpha) * __half2float(y[row_inA]) * __half2float(x[col_inA]));
}

#endif