#include <iostream>
#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>
#include <thrust/complex.h>

constexpr int BLOCK_SIZE = 256;

template <unsigned int blockSize, typename T>
__device__ void warpReduce(/*volatile*/ T *sdata, unsigned int tid) {
    if (blockSize >= 64) {sdata[tid] += sdata[tid + 32]; __syncwarp();}
    if (blockSize >= 32) {sdata[tid] += sdata[tid + 16]; __syncwarp();}
    if (blockSize >= 16) {sdata[tid] += sdata[tid + 8]; __syncwarp();}
    if (blockSize >= 8) {sdata[tid] += sdata[tid + 4]; __syncwarp();}
    if (blockSize >= 4) {sdata[tid] += sdata[tid + 2]; __syncwarp();}
    if (blockSize >= 2) {sdata[tid] += sdata[tid + 1]; __syncwarp();}
}

template <unsigned int blockSize, typename T>
__global__ void reduce(T *g_idata, T *g_odata, unsigned int n) {
    extern __shared__ char smem[];
    T * sdata = reinterpret_cast<T *>(smem);
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*(blockSize*2) + tid;
    unsigned int gridSize = blockSize*2*gridDim.x;
    sdata[tid] = 0;
    while (i < n) { 
        sdata[tid] += g_idata[i] + (i+blockSize < n ? g_idata[i+blockSize] : T(0));
        i += gridSize; 
    }
    __syncthreads();
    if (blockSize >= 512) { 
        if (tid < 256) { 
            sdata[tid] += sdata[tid + 256]; 
        } 
        __syncthreads(); 
    }
    if (blockSize >= 256) { 
        if (tid < 128) { 
            sdata[tid] += sdata[tid + 128]; 
        } 
        __syncthreads(); 
    }
    if (blockSize >= 128) { 
        if (tid < 64) { 
            sdata[tid] += sdata[tid + 64]; 
        } 
        __syncthreads(); 
    }
    if (tid < 32) warpReduce<BLOCK_SIZE, T>(sdata, tid);
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}


template <typename T>
__global__ void axpy_kernel(int n, T *x, T *y, T alpha){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n){
        y[i] = alpha * x[i] + y[i];
    }
}

template <typename T>
void axpy(unsigned int n, T *x, T *y, T alpha){
    unsigned int numblocks = (n + BLOCK_SIZE-1) / BLOCK_SIZE;
    T *gpux, *gpuy;
    unsigned int size = n * sizeof(T);
    cudaMalloc(&gpux, size);
    cudaMalloc(&gpuy, size);
    cudaMemcpy(gpux, x, size, cudaMemcpyHostToDevice);
    cudaMemcpy(gpuy, y, size, cudaMemcpyHostToDevice);
    axpy_kernel<<<numblocks, BLOCK_SIZE>>>(n, gpux, gpuy, alpha);
    cudaError_t err = cudaGetLastError();
    std::cout<<cudaGetErrorString(err)<<"\n";
    cudaMemcpy(x, gpux, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(y, gpuy, size, cudaMemcpyDeviceToHost);
    cudaFree(gpux);
    cudaFree(gpuy);
}

template <typename T>
__global__ void scal_kernel(int n, T *x, T alpha) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n){
        x[i] = alpha * x[i];
    }
}

template <typename T>
void scal(unsigned int n, T *x, T alpha){
    unsigned int numblocks = (n + BLOCK_SIZE-1) / BLOCK_SIZE;
    T *gpux;
    unsigned int size = n * sizeof(T);
    cudaMalloc(&gpux, size);
    cudaMemcpy(gpux, x, size, cudaMemcpyHostToDevice);
    scal_kernel<<<numblocks, BLOCK_SIZE>>>(n, gpux, alpha);
    cudaError_t err = cudaGetLastError();
    std::cout<<cudaGetErrorString(err)<<"\n";
    cudaMemcpy(x, gpux, size, cudaMemcpyDeviceToHost);
    cudaFree(gpux);
}

template <typename T>
__global__ void copy_kernel(int n, T *x, T *y){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n){
        y[i] = x[i];
    }
}

template <typename T>
void copy(unsigned int n, T *x, T *y){
    unsigned int numblocks = (n + BLOCK_SIZE-1) / BLOCK_SIZE;
    T *gpux, *gpuy;
    unsigned int size = n * sizeof(T);
    cudaMalloc(&gpux, size);
    cudaMalloc(&gpuy, size);
    cudaMemcpy(gpux, x, size, cudaMemcpyHostToDevice);
    cudaMemcpy(gpuy, y, size, cudaMemcpyHostToDevice);
    copy_kernel<<<numblocks, BLOCK_SIZE>>>(n, gpux, gpuy);
    cudaError_t err = cudaGetLastError();
    std::cout<<cudaGetErrorString(err)<<"\n";
    cudaMemcpy(y, gpuy, size, cudaMemcpyDeviceToHost);
    cudaFree(gpux);
    cudaFree(gpuy);
}

template <typename T>
__global__ void swap_kernel(int n, T *x, T *y){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n){
        T temp = x[i];
        x[i] = y[i];
        y[i] = temp;
    }
}

template <typename T>
void swap(unsigned int n, T *x, T *y){
    unsigned int numblocks = (n + BLOCK_SIZE-1) / BLOCK_SIZE;
    T *gpux, *gpuy;
    unsigned int size = n * sizeof(T);
    cudaMalloc(&gpux, size);
    cudaMalloc(&gpuy, size);
    cudaMemcpy(gpux, x, size, cudaMemcpyHostToDevice);
    cudaMemcpy(gpuy, y, size, cudaMemcpyHostToDevice);
    swap_kernel<<<numblocks, BLOCK_SIZE>>>(n, gpux, gpuy);
    cudaError_t err = cudaGetLastError();
    std::cout<<cudaGetErrorString(err)<<"\n";
    cudaMemcpy(x, gpux, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(y, gpuy, size, cudaMemcpyDeviceToHost);
    cudaFree(gpux);
    cudaFree(gpuy);
}

template <typename T>
__global__ void single_dot(int n, T *x, T *y, T *res){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n){
        res[i] = x[i] * y[i];
    }
}

template <typename T>
__global__ void single_dotc(int n, thrust::complex<T> *x, thrust::complex<T> *y, thrust::complex<T> *res){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n){
        res[i] = thrust::conj(x[i]) * y[i];
    }
}

template <typename T>
T dot(unsigned int n, T *x, T *y){
    T result = T(0);
    int numblocks = (n + BLOCK_SIZE-1)/BLOCK_SIZE;
    T *gpua, *gpub;
    T *gpu_intermediate_res;
    T *gpu_res, *res = new T[numblocks];
    int size = n * sizeof(T);
    // float *res;
    cudaMalloc(&gpua, size);
    cudaMalloc(&gpub, size);
    cudaMalloc(&gpu_intermediate_res, size);
    cudaMalloc(&gpu_res, numblocks * sizeof(T));
    cudaMemcpy (gpua, x, size, cudaMemcpyHostToDevice);
    cudaMemcpy (gpub, y, size, cudaMemcpyHostToDevice);
    single_dot<<<numblocks, BLOCK_SIZE>>>(n, gpua, gpub, gpu_intermediate_res);
    cudaError_t err = cudaGetLastError();
    std::cout<<cudaGetErrorString(err)<<"\n";
    reduce<BLOCK_SIZE, T><<<numblocks, BLOCK_SIZE, BLOCK_SIZE * sizeof(T)>>>(gpu_intermediate_res, gpu_res, n);
    cudaError_t err2 = cudaGetLastError();
    std::cout<<cudaGetErrorString(err2)<<"\n";
    cudaMemcpy (res, gpu_res, numblocks * sizeof(T), cudaMemcpyDeviceToHost);
    for(int i = 0; i < numblocks; ++i){
        result += res[i];
    }
    cudaFree(gpua);
    cudaFree(gpub);
    cudaFree(gpu_res);
    cudaFree(gpu_intermediate_res);
    delete[] res;
    return result;
}

template <typename T>
thrust::complex<T> dotu(unsigned int n, thrust::complex<T> *x, thrust::complex<T> *y){
    thrust::complex<T> result = thrust::complex<T>(T(0), T(0));
    int numblocks = (n + BLOCK_SIZE-1)/BLOCK_SIZE;
    thrust::complex<T> *gpua, *gpub;
    thrust::complex<T> *gpu_intermediate_res;
    thrust::complex<T> *gpu_res, *res = new thrust::complex<T>[numblocks];
    int size = n * sizeof(thrust::complex<T>);
    // float *res;
    cudaMalloc(&gpua, size);
    cudaMalloc(&gpub, size);
    cudaMalloc(&gpu_intermediate_res, size);
    cudaMalloc(&gpu_res, numblocks * sizeof(thrust::complex<T>));
    cudaMemcpy (gpua, x, size, cudaMemcpyHostToDevice);
    cudaMemcpy (gpub, y, size, cudaMemcpyHostToDevice);
    single_dot<thrust::complex<T>><<<numblocks, BLOCK_SIZE>>>(n, gpua, gpub, gpu_intermediate_res);
    cudaError_t err = cudaGetLastError();
    std::cout<<cudaGetErrorString(err)<<"\n";
    reduce<BLOCK_SIZE, thrust::complex<T>><<<numblocks, BLOCK_SIZE, BLOCK_SIZE * sizeof(thrust::complex<T>)>>>(gpu_intermediate_res, gpu_res, n);
    cudaError_t err2 = cudaGetLastError();
    std::cout<<cudaGetErrorString(err2)<<"\n";
    cudaMemcpy (res, gpu_res, numblocks * sizeof(thrust::complex<T>), cudaMemcpyDeviceToHost);
    for(int i = 0; i < numblocks; ++i){
        result += res[i];
    }
    cudaFree(gpua);
    cudaFree(gpub);
    cudaFree(gpu_res);
    cudaFree(gpu_intermediate_res);
    delete[] res;
    return result;
}

template <typename T>
thrust::complex<T> dotc(unsigned int n, thrust::complex<T> *x, thrust::complex<T> *y){
    thrust::complex<T> result = thrust::complex<T>(T(0), T(0));
    int numblocks = (n + BLOCK_SIZE-1)/BLOCK_SIZE;
    thrust::complex<T> *gpua, *gpub;
    thrust::complex<T> *gpu_intermediate_res;
    thrust::complex<T> *gpu_res, *res = new thrust::complex<T>[numblocks];
    int size = n * sizeof(thrust::complex<T>);
    // float *res;
    cudaMalloc(&gpua, size);
    cudaMalloc(&gpub, size);
    cudaMalloc(&gpu_intermediate_res, size);
    cudaMalloc(&gpu_res, numblocks * sizeof(thrust::complex<T>));
    cudaMemcpy (gpua, x, size, cudaMemcpyHostToDevice);
    cudaMemcpy (gpub, y, size, cudaMemcpyHostToDevice);
    single_dotc<<<numblocks, BLOCK_SIZE>>>(n, gpua, gpub, gpu_intermediate_res);
    cudaError_t err = cudaGetLastError();
    std::cout<<cudaGetErrorString(err)<<"\n";
    reduce<BLOCK_SIZE, thrust::complex<T>><<<numblocks, BLOCK_SIZE, BLOCK_SIZE * sizeof(thrust::complex<T>)>>>(gpu_intermediate_res, gpu_res, n);
    cudaError_t err2 = cudaGetLastError();
    std::cout<<cudaGetErrorString(err2)<<"\n";
    cudaMemcpy (res, gpu_res, numblocks * sizeof(thrust::complex<T>), cudaMemcpyDeviceToHost);
    for(int i = 0; i < numblocks; ++i){
        result += res[i];
    }
    cudaFree(gpua);
    cudaFree(gpub);
    cudaFree(gpu_res);
    cudaFree(gpu_intermediate_res);
    delete[] res;
    return result;
}

template <typename T>
T dot_double_precision(unsigned int n, T *x, T *y){
    double result = 0.0;
    double *doublex = new double[n], *doubley = new double[n];
    for(int i = 0; i < n; ++i){
        doublex[i] = x[i];
        doubley[i] = y[i];
    }
    int numblocks = (n + BLOCK_SIZE-1)/BLOCK_SIZE;
    double *gpua, *gpub;
    double *gpu_intermediate_res;
    double *gpu_res, *res = new double[numblocks];
    int size = n * sizeof(double);
    // float *res;
    cudaMalloc(&gpua, size);
    cudaMalloc(&gpub, size);
    cudaMalloc(&gpu_intermediate_res, size);
    cudaMalloc(&gpu_res, numblocks * sizeof(double));
    cudaMemcpy (gpua, doublex, size, cudaMemcpyHostToDevice);
    cudaMemcpy (gpub, doubley, size, cudaMemcpyHostToDevice);
    single_dot<<<numblocks, BLOCK_SIZE>>>(n, gpua, gpub, gpu_intermediate_res);
    cudaError_t err = cudaGetLastError();
    std::cout<<cudaGetErrorString(err)<<"\n";
    reduce<BLOCK_SIZE, double><<<numblocks, BLOCK_SIZE, BLOCK_SIZE * sizeof(double)>>>(gpu_intermediate_res, gpu_res, n);
    cudaError_t err2 = cudaGetLastError();
    std::cout<<cudaGetErrorString(err2)<<"\n";
    cudaMemcpy (res, gpu_res, numblocks * sizeof(double), cudaMemcpyDeviceToHost);
    for(int i = 0; i < numblocks; ++i){
        result += res[i];
    }
    cudaFree(gpua);
    cudaFree(gpub);
    cudaFree(gpu_res);
    cudaFree(gpu_intermediate_res);
    delete[] res;
    delete[] doublex;
    delete[] doubley;
    return static_cast<T>(result);
}

template <typename T>
T nrm2(unsigned int n, T *x){
    T norm = sqrt(dot(n, x, x));
    return norm;
}

template <typename T>
__global__ void absolute(int n, thrust::complex<T> *x, T *y){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n){
        y[i] = abs(x[i].real()) + abs(x[i].imag());
    }
}

template <typename T>
T asum(unsigned int n, thrust::complex<T> *x){
    T result = T(0);
    unsigned int numblocks = (n + BLOCK_SIZE-1) / BLOCK_SIZE;
    thrust::complex<T> *gpua;
    T *gpu_intermediate_res;
    T *gpu_res, *res = new T[numblocks];
    int size = sizeof(thrust::complex<T>) * n;
    cudaMalloc(&gpua, size);
    cudaMalloc(&gpu_res, sizeof(T) * numblocks);
    cudaMalloc(&gpu_intermediate_res, sizeof(T) * n);
    cudaMemcpy(gpua, x, size, cudaMemcpyHostToDevice);
    absolute<<<numblocks, BLOCK_SIZE>>>(n, gpua, gpu_intermediate_res);
    cudaError_t err = cudaGetLastError();
    std::cout<<cudaGetErrorString(err)<<"\n";
    reduce<BLOCK_SIZE, T><<<numblocks, BLOCK_SIZE, BLOCK_SIZE * sizeof(T)>>>(gpu_intermediate_res, gpu_res, n);
    cudaError_t err2 = cudaGetLastError();
    std::cout<<cudaGetErrorString(err2)<<"\n";
    cudaMemcpy(res, gpu_res, sizeof(T)*numblocks, cudaMemcpyDeviceToHost);
    for(int i = 0; i < numblocks; ++i){
        result += res[i];
    }
    delete[] res;
    cudaFree(gpua);
    cudaFree(gpu_intermediate_res);
    cudaFree(gpu_res);
    return result;
}

template <typename T>
int i_amax(unsigned int n, thrust::complex<T> *x){
    unsigned int numblocks = (n + BLOCK_SIZE-1) / BLOCK_SIZE;
    thrust::complex<T> *gpua;
    T *gpu_intermediate_res;
    T *res = new T[n];
    int size = sizeof(thrust::complex<T>) * n;
    cudaMalloc(&gpua, size);
    cudaMalloc(&gpu_intermediate_res, sizeof(T) * n);
    cudaMemcpy(gpua, x, size, cudaMemcpyHostToDevice);
    absolute<<<numblocks, BLOCK_SIZE>>>(n, gpua, gpu_intermediate_res);
    cudaError_t err = cudaGetLastError();
    std::cout<<cudaGetErrorString(err)<<"\n";
    cudaMemcpy(res, gpu_intermediate_res, sizeof(T) * n, cudaMemcpyDeviceToHost);
    int result = 0;
    T curmax = T(0);
    for(int i = 0; i < n; ++i){
        if(res[i] > curmax){
            curmax = res[i];
            result = i;
        }
    }
    delete[] res;
    cudaFree(gpua);
    cudaFree(gpu_intermediate_res);
    return result;
}


template <typename T, typename typeC, typename typeS>
void rotg(T& a, T& b, typeC& c, typeS& s){
    T legacya = a, legacyb = b;
    T r = sqrt(a*a + b*b);
    c = a/r;
    s = b/r;
    a = r;
    if(abs(legacya) > abs(legacyb)) {
        b = s;
    }else if(abs(legacya) <= abs(legacyb) && c != typeC(0)){
        b = typeC(1)/c;
    }else if(c == typeC(0)) b = T(1);
}

template <typename T, typename typeC, typename typeS>
__global__ void rot_kernel(unsigned int n, T *x, T *y, typeC c, typeS s){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n){
        T legacyx = x[i];
        x[i] = x[i] * c - y[i] * s;
        y[i] = legacyx * s + y[i] * c;
    }
}

template <typename T>
void rotmg(T *d1, T *d2, T *x1, const T *y1, T *param) {

}

template <typename T, typename typeC, typename typeS>
void rot(unsigned int n, T *x, T *y, typeC c, typeS s){
    unsigned int numblocks = (n + BLOCK_SIZE-1) / BLOCK_SIZE;
    T *gpux, *gpuy;
    unsigned int size = n * sizeof(T);
    cudaMalloc(&gpux, size);
    cudaMalloc(&gpuy, size);
    cudaMemcpy(gpux, x, size, cudaMemcpyHostToDevice);
    cudaMemcpy(gpuy, y, size, cudaMemcpyHostToDevice);
    rot_kernel<T, typeC, typeS><<<numblocks, BLOCK_SIZE>>>(n, gpux, gpuy, c, s);
    cudaError_t err = cudaGetLastError();
    std::cout<<cudaGetErrorString(err)<<"\n";
    cudaMemcpy(x, gpux, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(y, gpuy, size, cudaMemcpyDeviceToHost);
    cudaFree(gpux);
    cudaFree(gpuy);
}

int main(){
    thrust::complex<float> m = thrust::complex<float>(1.0f, 1.0f);
    thrust::complex<float> n = thrust::complex<float> (2.0f, 1.0f);
    constexpr unsigned int N = 5;
    thrust::complex<float> *a = new thrust::complex<float>[N];
    thrust::complex<float> *b = new thrust::complex<float>[N];
    thrust::complex<float> *f = new thrust::complex<float>[2];
    f[0] = m, f[1] = n;
    float *x = new float[N];
    float *y = new float[N];
    for(int i = 0; i < N; ++i){
        a[i] = m;
        b[i] = n;
        x[i] = 1.0f;
        y[i] = 2.0f;
    }
    thrust::complex<float> res = dotc(N, a, b);
    float res2 = dot_double_precision(N, x, y);
    int res3 = i_amax(2, f);
    std::cout<<res<<"\n";
    std::cout<<res2<<"\n";
    std::cout<<res3<<"\n";
    delete[] a;
    delete[] b;
    delete[] x;
    delete[] y;
    delete[] f;
    return 0;
}