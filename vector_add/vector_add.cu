#include <cuda_runtime.h>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>

// 简单的 CUDA 错误检查辅助宏。
// 在学习代码里，尽早失败可以快速定位问题。
#define CUDA_CHECK(call)                                                            \
    do {                                                                            \
        cudaError_t err__ = (call);                                                 \
        if (err__ != cudaSuccess) {                                                 \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__         \
                      << " code=" << static_cast<int>(err__)                       \
                      << " (" << cudaGetErrorName(err__) << ")"                   \
                      << " message=" << cudaGetErrorString(err__) << std::endl;    \
            std::exit(EXIT_FAILURE);                                                \
        }                                                                           \
    } while (0)

// Kernel: C = A + B
// 这里使用 grid-stride loop（网格步进循环），是工程中常见写法。
// 重要性：
// 1) 对任意长度 N 都通用，即使 N 远大于一次启动的总线程数。
// 2) 每个线程可以按步长处理多个元素。
__global__ void vectorAddKernel(const float* a, const float* b, float* c, int n) {
    int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = global_tid; i < n; i += stride) {
        c[i] = a[i] + b[i];
    }
}

// CPU 参考实现。
// 每次优化 GPU 代码后，都可用 CPU 结果做正确性校验。
void vectorAddCPU(const std::vector<float>& a,
                  const std::vector<float>& b,
                  std::vector<float>& c) {
    for (size_t i = 0; i < a.size(); ++i) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    // 第一课的问题规模。
    // 1<<20 = 1,048,576 个元素。
    const int n = 1 << 20;
    const size_t bytes = static_cast<size_t>(n) * sizeof(float);

    std::cout << "[Stage A - CUDA Basics] Vector Add" << std::endl;
    std::cout << "Elements: " << n << ", bytes per array: " << bytes << std::endl;

    // Host 侧缓冲区（CPU 内存）。
    std::vector<float> h_a(n), h_b(n), h_c_gpu(n), h_c_cpu(n);

    // 初始化输入数据。
    // 使用确定性模式，便于调试与复现。
    for (int i = 0; i < n; ++i) {
        h_a[i] = 0.001f * static_cast<float>(i);
        h_b[i] = 1.0f + 0.002f * static_cast<float>(i);
    }

    // Device 侧缓冲区（GPU 显存）。
    float* d_a = nullptr;
    float* d_b = nullptr;
    float* d_c = nullptr;

    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));

    // 将输入数据从 CPU 拷贝到 GPU。
    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice));

    // Kernel 启动配置。
    // 实战中 threads_per_block 常见为 128/256/512；这里先用 256 作为默认值。
    const int threads_per_block = 256;

    // 理论上覆盖一次 N 所需 block 数。
    // 由于用了 grid-stride loop，N 变大时依然健壮。
    int blocks = (n + threads_per_block - 1) / threads_per_block;

    // 可选上限，避免简单算子启动过多小 block。
    // 这样在不同 GPU 上演示行为更稳定。
    blocks = std::min(blocks, 4096);

    std::cout << "Launch config: blocks=" << blocks
              << ", threads_per_block=" << threads_per_block << std::endl;

    // 使用 CUDA Event 对 Kernel 计时。
    // 这里只统计 Kernel 执行时间（不含 memcpy），便于观察纯计算行为。
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    vectorAddKernel<<<blocks, threads_per_block>>>(d_a, d_b, d_c, n);
    CUDA_CHECK(cudaEventRecord(stop));

    // 检查异步 Kernel 启动错误。
    CUDA_CHECK(cudaGetLastError());

    // 等待 Kernel 完成，再读取时间与输出。
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    // 将结果从 GPU 拷回 CPU。
    CUDA_CHECK(cudaMemcpy(h_c_gpu.data(), d_c, bytes, cudaMemcpyDeviceToHost));

    // CPU 基线结果，用于正确性对照。
    vectorAddCPU(h_a, h_b, h_c_cpu);

    // 结果校验。
    // 浮点计算通常采用误差阈值，而不是直接做“完全相等”比较。
    const float eps = 1e-5f;
    int mismatches = 0;
    float max_abs_err = 0.0f;
    int max_err_idx = -1;

    for (int i = 0; i < n; ++i) {
        float err = std::fabs(h_c_gpu[i] - h_c_cpu[i]);
        if (err > eps) {
            ++mismatches;
            if (mismatches <= 5) {
                std::cout << "Mismatch at i=" << i
                          << ": gpu=" << h_c_gpu[i]
                          << ", cpu=" << h_c_cpu[i]
                          << ", abs_err=" << err << std::endl;
            }
        }
        if (err > max_abs_err) {
            max_abs_err = err;
            max_err_idx = i;
        }
    }

    // 有效内存吞吐率估算（粗略，用于学习理解）。
    // 2 次读 + 1 次写 = 3 * N * sizeof(float) 字节搬运量。
    const double moved_bytes = 3.0 * static_cast<double>(bytes);
    const double gb_per_s = (moved_bytes / 1e9) / (static_cast<double>(ms) / 1e3);

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Kernel time: " << ms << " ms" << std::endl;
    std::cout << "Approx throughput: " << gb_per_s << " GB/s" << std::endl;
    std::cout << "Max abs error: " << max_abs_err
              << " (index " << max_err_idx << ")" << std::endl;

    if (mismatches == 0) {
        std::cout << "Result check: PASS" << std::endl;
    } else {
        std::cout << "Result check: FAIL, mismatches=" << mismatches << std::endl;
    }

    // 资源清理。
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));

    return (mismatches == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}
