#include <stdio.h>

// a simple CUDA kernel function that prints "Hello World" from the GPU
__global__ void hello_world() {
    printf("GPU: Hello World\n");
}

int main() {
    printf("CPU: Hello CUDA\n");
    hello_world<<<1, 10>>>();
    cudaDeviceReset(); // if no this line ,it can not output hello world from gpu
    return 0;
}