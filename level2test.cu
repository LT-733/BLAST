#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "level2.cuh"

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "[CUDA ERROR] at " << __FILE__ << ":" << __LINE__ \
                      << " -> " << cudaGetErrorString(err) << std::endl; \
            return false; \
        } \
    } while (0)

bool close_enough_rel(float a, float b, float rel_tol = 5e-2f) {
    float diff = std::fabs(a - b);
    if (diff < 1e-2f) return true;
    return (diff / std::max(std::fabs(a), std::fabs(b))) < rel_tol;
}

// 1. GEMV (M = 1048576, N = 16 -> Tall-and-Skinny Matrix)
bool test_gemv_large() {
    std::cout << "[Running GEMV Large Test (M=1048576, N=16)]..." << std::endl;
    constexpr unsigned int m = 1 << 20, n = 16;
    size_t size_A = static_cast<size_t>(m) * n;

    // Use clean power-of-two values exact in FP16 (e.g., 0.25f)
    std::vector<half> h_A(size_A, __float2half(0.25f));
    std::vector<half> h_x(n, __float2half(1.0f));
    std::vector<half> h_y(m, __float2half(2.0f));
    float alpha_f = 1.0f, beta_f = 0.5f;

    half *d_A, *d_x, *d_y;
    CUDA_CHECK(cudaMalloc(&d_A, size_A * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_x, n * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_y, m * sizeof(half)));

    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), size_A * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), n * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y, h_y.data(), m * sizeof(half), cudaMemcpyHostToDevice));

    dim3 block(BLOCK_SIZE), grid((m + WARP_PER_BLOCK - 1) / WARP_PER_BLOCK);
    gemv_kernel<<<grid, block>>>(m, n, d_A, d_x, d_y, __float2half(alpha_f), __float2half(beta_f));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_y.data(), d_y, m * sizeof(half), cudaMemcpyDeviceToHost));

    // 16 * 0.25 * 1.0 = 4.0; alpha*4.0 + beta*2.0 = 5.0
    float expected_val = alpha_f * (n * 0.25f * 1.0f) + beta_f * 2.0f;
    bool passed = true;
    for (size_t i = 0; i < m; i += m / 10) {
        float res = __half2float(h_y[i]);
        if (!close_enough_rel(res, expected_val)) {
            std::cerr << "  [FAIL] Index " << i << " expected " << expected_val << " got " << res << std::endl;
            passed = false;
        }
    }
    cudaFree(d_A); cudaFree(d_x); cudaFree(d_y);
    if (passed) std::cout << "  -> GEMV Large Passed!" << std::endl;
    return passed;
}

// 2. TRMV Naive (N = 1024 -> 1,048,576 total matrix elements)
bool test_trmv_large() {
    std::cout << "[Running TRMV Naive Large Test (N=1024)]..." << std::endl;
    constexpr unsigned int n = 1024;
    size_t size_A = static_cast<size_t>(n) * n;

    std::vector<half> h_A(size_A, __float2half(0.0f));
    // Fill upper triangle with 0.125f
    for (unsigned int i = 0; i < n; ++i) {
        for (unsigned int j = i; j < n; ++j) {
            h_A[i * n + j] = __float2half(0.125f);
        }
    }
    std::vector<half> h_x(n, __float2half(1.0f));
    std::vector<half> h_res(n);

    half *d_A, *d_x, *d_res;
    CUDA_CHECK(cudaMalloc(&d_A, size_A * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_x, n * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_res, n * sizeof(half)));

    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), size_A * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), n * sizeof(half), cudaMemcpyHostToDevice));

    trmv<<<(n + 255) / 256, 256>>>(n, 'U', d_A, d_x, d_res);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_res.data(), d_res, n * sizeof(half), cudaMemcpyDeviceToHost));

    bool passed = true;
    for (size_t i = 0; i < n; i += n / 10) {
        float expected_val = (n - i) * 0.125f * 1.0f;
        float res = __half2float(h_res[i]);
        if (!close_enough_rel(res, expected_val)) {
            std::cerr << "  [FAIL] Index " << i << " expected " << expected_val << " got " << res << std::endl;
            passed = false;
        }
    }
    cudaFree(d_A); cudaFree(d_x); cudaFree(d_res);
    if (passed) std::cout << "  -> TRMV Naive Large Passed!" << std::endl;
    return passed;
}

// 3. TRMV Optimized (N = 1024 -> 1,048,576 total matrix elements)
bool test_trmv_optimized_large() {
    std::cout << "[Running TRMV Optimized Large Test (N=1024)]..." << std::endl;
    constexpr unsigned int n = 1024;
    size_t size_A = static_cast<size_t>(n) * n;

    std::vector<half> h_A(size_A, __float2half(0.0f));
    for (unsigned int i = 0; i < n; ++i) {
        for (unsigned int j = i; j < n; ++j) {
            h_A[i * n + j] = __float2half(0.125f);
        }
    }
    std::vector<half> h_x(n, __float2half(1.0f));
    std::vector<half> h_res(n);

    half *d_A, *d_x, *d_res;
    CUDA_CHECK(cudaMalloc(&d_A, size_A * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_x, n * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_res, n * sizeof(half)));

    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), size_A * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), n * sizeof(half), cudaMemcpyHostToDevice));

    dim3 block(BLOCK_SIZE), grid((n + WARP_PER_BLOCK - 1) / WARP_PER_BLOCK);
    trmv_optimized<<<grid, block>>>(n, 'U', d_A, d_x, d_res);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_res.data(), d_res, n * sizeof(half), cudaMemcpyDeviceToHost));

    bool passed = true;
    for (size_t i = 0; i < n; i += n / 10) {
        float expected_val = (n - i) * 0.125f * 1.0f;
        float res = __half2float(h_res[i]);
        if (!close_enough_rel(res, expected_val)) {
            std::cerr << "  [FAIL] Index " << i << " expected " << expected_val << " got " << res << std::endl;
            passed = false;
        }
    }
    cudaFree(d_A); cudaFree(d_x); cudaFree(d_res);
    if (passed) std::cout << "  -> TRMV Optimized Large Passed!" << std::endl;
    return passed;
}

// 4. GER (M = 1048576, N = 2 -> A is M x N = 2,097,152 elements)
bool test_ger_large() {
    std::cout << "[Running GER Large Test (M=1048576, N=2)]..." << std::endl;
    constexpr unsigned int m = 1 << 20, n = 2;
    size_t size_A = static_cast<size_t>(m) * n;

    std::vector<half> h_A(size_A, __float2half(0.0f));
    std::vector<half> h_x(m, __float2half(1.5f));
    std::vector<half> h_y(n, __float2half(2.0f));
    float alpha = 0.5f;

    half *d_A, *d_x, *d_y;
    CUDA_CHECK(cudaMalloc(&d_A, size_A * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_x, m * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_y, n * sizeof(half)));

    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), size_A * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), m * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y, h_y.data(), n * sizeof(half), cudaMemcpyHostToDevice));

    ger<<<(size_A + 255) / 256, 256>>>(m, n, d_A, d_x, d_y, __float2half(alpha));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_A.data(), d_A, size_A * sizeof(half), cudaMemcpyDeviceToHost));

    float expected_val = alpha * 1.5f * 2.0f;
    bool passed = true;
    for (size_t i = 0; i < size_A; i += size_A / 10) {
        float res = __half2float(h_A[i]);
        if (!close_enough_rel(res, expected_val)) {
            std::cerr << "  [FAIL] Index " << i << " expected " << expected_val << " got " << res << std::endl;
            passed = false;
        }
    }
    cudaFree(d_A); cudaFree(d_x); cudaFree(d_y);
    if (passed) std::cout << "  -> GER Large Passed!" << std::endl;
    return passed;
}

// 5. SYR (N = 1024 -> A is 1024 x 1024 = 1,048,576 elements)
bool test_syr_large() {
    std::cout << "[Running SYR Large Test (N=1024)]..." << std::endl;
    constexpr unsigned int n = 1024;
    size_t size_A = static_cast<size_t>(n) * n;

    std::vector<half> h_A(size_A, __float2half(0.0f));
    std::vector<half> h_x(n, __float2half(2.0f));
    float alpha = 0.5f;

    half *d_A, *d_x;
    CUDA_CHECK(cudaMalloc(&d_A, size_A * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_x, n * sizeof(half)));

    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), size_A * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), n * sizeof(half), cudaMemcpyHostToDevice));

    syr<<<(size_A + 255) / 256, 256>>>(n, 'L', d_A, d_x, __float2half(alpha));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_A.data(), d_A, size_A * sizeof(half), cudaMemcpyDeviceToHost));

    float expected_val = alpha * (2.0f * 2.0f);
    bool passed = true;
    // Check lower-triangular values
    for (size_t i = 0; i < n; i += n / 10) {
        size_t idx = i * n + 0; // Column 0 of row i (lower triangle)
        float res = __half2float(h_A[idx]);
        if (!close_enough_rel(res, expected_val)) {
            std::cerr << "  [FAIL] Index " << idx << " expected " << expected_val << " got " << res << std::endl;
            passed = false;
        }
    }
    cudaFree(d_A); cudaFree(d_x);
    if (passed) std::cout << "  -> SYR Large Passed!" << std::endl;
    return passed;
}

// 6. SYR2 (N = 1024 -> A is 1024 x 1024 = 1,048,576 elements)
bool test_syr2_large() {
    std::cout << "[Running SYR2 Large Test (N=1024)]..." << std::endl;
    constexpr unsigned int n = 1024;
    size_t size_A = static_cast<size_t>(n) * n;

    std::vector<half> h_A(size_A, __float2half(0.0f));
    std::vector<half> h_x(n, __float2half(1.5f));
    std::vector<half> h_y(n, __float2half(2.0f));
    float alpha = 0.5f;

    half *d_A, *d_x, *d_y;
    CUDA_CHECK(cudaMalloc(&d_A, size_A * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_x, n * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_y, n * sizeof(half)));

    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), size_A * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), n * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y, h_y.data(), n * sizeof(half), cudaMemcpyHostToDevice));

    syr2<<<(size_A + 255) / 256, 256>>>(n, 'L', d_A, d_x, d_y, __float2half(alpha));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_A.data(), d_A, size_A * sizeof(half), cudaMemcpyDeviceToHost));

    float expected_val = alpha * (1.5f * 2.0f + 2.0f * 1.5f);
    bool passed = true;
    for (size_t i = 0; i < n; i += n / 10) {
        size_t idx = i * n + 0;
        float res = __half2float(h_A[idx]);
        if (!close_enough_rel(res, expected_val)) {
            std::cerr << "  [FAIL] Index " << idx << " expected " << expected_val << " got " << res << std::endl;
            passed = false;
        }
    }
    cudaFree(d_A); cudaFree(d_x); cudaFree(d_y);
    if (passed) std::cout << "  -> SYR2 Large Passed!" << std::endl;
    return passed;
}

int main() {
    int passed_count = 0;
    int total_tests = 6;

    if (test_gemv_large()) passed_count++;
    if (test_trmv_large()) passed_count++;
    if (test_trmv_optimized_large()) passed_count++;
    if (test_ger_large()) passed_count++;
    if (test_syr_large()) passed_count++;
    if (test_syr2_large()) passed_count++;

    std::cout << "\n==========================================" << std::endl;
    std::cout << "Large-Scale Test Summary: " << passed_count << "/" << total_tests << " passed." << std::endl;
    std::cout << "==========================================" << std::endl;

    return 0;
}
