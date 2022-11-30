#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define DataType double

__device__ DataType add(DataType a, DataType b) {
    return a + b;
}

__global__ void vecAdd(DataType *in1, DataType *in2, DataType *out, int len) {
    //@@ Insert code to implement vector addition here
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= len) return;

    out[idx] = add(in1[idx], in2[idx]);
}

//@@ Insert code to implement timer start
//@@ Insert code to implement timer stop
double cpuSecond() {
   struct timeval tp;
   gettimeofday(&tp,NULL);
   return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
}

int main(int argc, char **argv) {
    int inputLength;
    DataType *hostInput1;
    DataType *hostInput2;
    DataType *hostOutput;
    DataType *resultRef;
    DataType *deviceInput1;
    DataType *deviceInput2;
    DataType *deviceOutput;

    //@@ Insert code below to read in inputLength from args
    inputLength = atoi(argv[1]);
    printf("The input length is %d.\n", inputLength);

    //@@ Insert code below to allocate Host memory for input and output
    hostInput1 = (DataType*) malloc(inputLength * sizeof(DataType));
    hostInput2 = (DataType*) malloc(inputLength * sizeof(DataType));
    hostOutput = (DataType*) malloc(inputLength * sizeof(DataType));
    resultRef = (DataType*) malloc(inputLength * sizeof(DataType));

    //@@ Insert code below to initialize hostInput1 and hostInput2 to random numbers, and create reference result in CPU
    for (int i = 0; i < inputLength; ++i) {
        DataType randomNumber1 = rand() / (DataType) RAND_MAX; // Random number in interval [0, 1.0]
        DataType randomNumber2 = rand() / (DataType) RAND_MAX; // Random number in interval [0, 1.0]
        hostInput1[i] = randomNumber1;
        hostInput2[i] = randomNumber2;
        resultRef[i] = randomNumber1 + randomNumber2;
    }

    //@@ Insert code below to allocate GPU memory here
    cudaMalloc(&deviceInput1, inputLength * sizeof(DataType));
    cudaMalloc(&deviceInput2, inputLength * sizeof(DataType));
    cudaMalloc(&deviceOutput, inputLength * sizeof(DataType));

    //@@ Insert code to below to Copy memory to the GPU here
    double start = cpuSecond();
    cudaMemcpy(deviceInput1, hostInput1, inputLength * sizeof(DataType), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceInput2, hostInput2, inputLength * sizeof(DataType), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    double duration = cpuSecond() - start;
    printf("Time Data copy (H2D): %f.\n", duration);

    //@@ Initialize the 1D grid and block dimensions here
    int Db = 128;
    int Dg = (inputLength + Db - 1) / Db;

    //@@ Launch the GPU Kernel here
    start = cpuSecond();
    vecAdd <<<Dg, Db>>>(deviceInput1, deviceInput2, deviceOutput, inputLength);
    cudaDeviceSynchronize();
    duration = cpuSecond() - start;
    printf("Time CUDA kernel: %f.\n", duration);

    //@@ Copy the GPU memory back to the CPU here
    start = cpuSecond();
    cudaMemcpy(hostOutput, deviceOutput, inputLength * sizeof(DataType), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    duration = cpuSecond() - start;
    printf("Time Data copy (D2H): %f.\n", duration);

    //@@ Insert code below to compare the output with the reference
    int equality = 1;
    for (int i = 0; i < inputLength; ++i) {
        //printf("CPU element: %f\n", resultRef[i]);
        //printf("GPU element: %f\n", hostOutput[i]);
        if (fabs(hostOutput[i] - resultRef[i]) > 1e-8) { // Compare if elements are approximately equal
            equality = 0;
            break;
        }
    }
    if (equality == 1) {
        printf("CPU and GPU results are equal.\n");
    } else {
        printf("CPU and GPU results are NOT equal.\n");
    }

    //@@ Free the GPU memory here
    cudaFree(deviceInput1);
    cudaFree(deviceInput2);
    cudaFree(deviceOutput);

    //@@ Free the CPU memory here
    free(hostInput1);
    free(hostInput2);
    free(hostOutput);
    free(resultRef);

    return 0;
}