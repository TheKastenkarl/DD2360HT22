#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define DataType double
#define NUMSTREAMS 4
#define DEBUG 0
//#define S_seg 1000000 // If this is not defined, then 'S_seg = inputLength / NUMSTREAMS' is used

__global__ void vecAdd(DataType *in1, DataType *in2, DataType *out, int segment_size, int offset) {
    // Insert code to implement vector addition here
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= segment_size) return; // equivalent to: offset + idx >= offset + segment_size

    #if DEBUG
    printf("offset + idx = %d\n", offset + idx);
    printf("in1[%d] = %f\n", offset + idx, in1[offset + idx]);
    printf("in2[%d] = %f\n", offset + idx, in2[offset + idx]);
    #endif
    out[offset + idx] = in1[offset + idx] + in2[offset + idx];
}

// Insert code to implement timer start
// Insert code to implement timer stop
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

    // Insert code below to read in inputLength from args
    inputLength = atoi(argv[1]);
    printf("The input length is %d.\n", inputLength);

    // Insert code below to allocate Host memory for input and output
    cudaHostAlloc(&hostInput1, inputLength * sizeof(DataType), cudaHostAllocDefault);
    cudaHostAlloc(&hostInput2, inputLength * sizeof(DataType), cudaHostAllocDefault);
    cudaHostAlloc(&hostOutput, inputLength * sizeof(DataType), cudaHostAllocDefault);
    cudaHostAlloc(&resultRef, inputLength * sizeof(DataType), cudaHostAllocDefault);

    // Insert code below to initialize hostInput1 and hostInput2 to random numbers, and create reference result in CPU
    for (int i = 0; i < inputLength; ++i) {
        DataType randomNumber1 = rand() / (DataType) RAND_MAX; // Random number in interval [0, 1.0]
        DataType randomNumber2 = rand() / (DataType) RAND_MAX; // Random number in interval [0, 1.0]
        hostInput1[i] = randomNumber1;
        hostInput2[i] = randomNumber2;
        #if DEBUG
        printf("hostInput1[%d] = %f\n", i, hostInput1[i]);
        printf("hostInput2[%d] = %f\n", i, hostInput2[i]);
        #endif
        resultRef[i] = randomNumber1 + randomNumber2;
    }

    // Allocate pinned GPU memory
    cudaMalloc(&deviceInput1, inputLength * sizeof(DataType));
    cudaMalloc(&deviceInput2, inputLength * sizeof(DataType));
    cudaMalloc(&deviceOutput, inputLength * sizeof(DataType));

    // Initialize CUDA streams
    #ifndef S_seg
    const int S_seg = inputLength / NUMSTREAMS;
    #endif
    const int numSegments = (int) ceil((float) inputLength / S_seg);
    #if DEBUG
    printf("numSegments = %d\n", numSegments);
    #endif
    const int segmentBytes = S_seg * sizeof(DataType);
    cudaStream_t stream[NUMSTREAMS];
    for (int i = 0; i < NUMSTREAMS; ++i) {
        cudaStreamCreate(&stream[i]);
    }

    // Initialize the 1D grid and block dimensions here
    int Db = 128;
    int Dg = (inputLength + Db - 1) / Db;

    // Start timer
    double start = cpuSecond();

    // Asynchronously copy the data and execute the kernel
    for (int i = 0; i < numSegments; ++i) {
        const int offset = i * S_seg;
        if (i < numSegments - 1) {
            cudaMemcpyAsync(&deviceInput1[offset], &hostInput1[offset], segmentBytes, cudaMemcpyHostToDevice, stream[i % NUMSTREAMS]);
            cudaMemcpyAsync(&deviceInput2[offset], &hostInput2[offset], segmentBytes, cudaMemcpyHostToDevice, stream[i % NUMSTREAMS]);
            vecAdd<<<Dg, Db, 0, stream[i % NUMSTREAMS]>>>(deviceInput1, deviceInput2, deviceOutput, S_seg, offset);
            cudaMemcpyAsync(&hostOutput[offset], &deviceOutput[offset], segmentBytes, cudaMemcpyDeviceToHost, stream[i % NUMSTREAMS]);
        } else {
            // Handle the case when 'inputLength' is not an integer multiple of 'numSegments'
            const int lastSegmentSize = inputLength - (numSegments - 1) * S_seg;
            const int lastSegmentBytes = lastSegmentSize * sizeof(DataType);
            #if DEBUG
            printf("lastSegmentSize = %d\n", lastSegmentSize);
            #endif
            cudaMemcpyAsync(&deviceInput1[offset], &hostInput1[offset], lastSegmentBytes, cudaMemcpyHostToDevice, stream[i % NUMSTREAMS]);
            cudaMemcpyAsync(&deviceInput2[offset], &hostInput2[offset], lastSegmentBytes, cudaMemcpyHostToDevice, stream[i % NUMSTREAMS]);
            vecAdd<<<Dg, Db, 0, stream[i % NUMSTREAMS]>>>(deviceInput1, deviceInput2, deviceOutput, lastSegmentSize, offset);
            cudaMemcpyAsync(&hostOutput[offset], &deviceOutput[offset], lastSegmentBytes, cudaMemcpyDeviceToHost, stream[i % NUMSTREAMS]);
        }
    }

    for (int i = 0; i < NUMSTREAMS; ++i) {
        cudaStreamDestroy(stream[i]);
    }

    cudaDeviceSynchronize();

    // Stop timer
    double duration = cpuSecond() - start;
    printf("Time Data copy (H2D + D2H) and CUDA kernel: %f.\n", duration);

    // Compare the output with the reference
    int equality = 1;
    for (int i = 0; i < inputLength; ++i) {
        #if DEBUG
        printf("hostOutput[%d] = %f\n", i, hostOutput[i]);
        printf("resultRef[%d] = %f\n", i, resultRef[i]);
        #endif
        if (fabs(hostOutput[i] - resultRef[i]) > 1e-8) { // Compare if elements are approximately equal
            equality = 0;
            #if !DEBUG
            break;
            #endif
        }
    }
    if (equality == 1) {
        printf("CPU and GPU results are equal.\n");
    } else {
        printf("CPU and GPU results are NOT equal.\n");
    }

    // Free the GPU memory here
    cudaFree(deviceInput1);
    cudaFree(deviceInput2);
    cudaFree(deviceOutput);

    // Free the CPU memory here
    cudaFreeHost(hostInput1);
    cudaFreeHost(hostInput2);
    cudaFreeHost(hostOutput);
    cudaFreeHost(resultRef);

    return 0;
}