# include <stdio.h>
# include <cuda_runtime.h>

// a simple CUDA kernel function that prints "Hello World" from the GPU
__global__ void hello_world() {
    printf("GPU: Hello World! I am thread %d\n", threadIdx.x);
}

int main() {
    printf("CPU: Hello CUDA\n");
    hello_world<<<1, 10>>>();
    cudaDeviceReset(); // if no this line ,it can not output hello world from gpu
    return 0;
}


// compile and run
// nvcc hello.cu -o hello | ./hello