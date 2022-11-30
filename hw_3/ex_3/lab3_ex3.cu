#include <stdio.h>
#include <sys/time.h>
#include <random>

#define NUM_BINS 4096

__global__ void histogram_kernel(unsigned int *input, unsigned int *bins,
                                 unsigned int num_elements,
                                 unsigned int num_bins) {
    //@@ Insert code below to compute histogram of input using shared memory and atomics
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elements) return;

    bins[input[idx]] += 1;
}

__global__ void convert_kernel(unsigned int *bins, unsigned int num_bins) {
    //@@ Insert code below to clean up bins that saturate at 127
    const int bin = blockIdx.x * blockDim.x + threadIdx.x;
    if (bin >= num_bins) return;

    if (bins[bin] > 127) {
        bins[bin] = 127;
    }
}

int main(int argc, char **argv) {
    int inputLength;
    unsigned int *hostInput;
    unsigned int *hostBins;
    unsigned int *resultRef;
    unsigned int *deviceInput;
    unsigned int *deviceBins;

    //@@ Insert code below to read in inputLength from args
    inputLength = atoi(argv[1]);
    printf("The input length is %d\n", inputLength);

    //@@ Insert code below to allocate Host memory for input and output
    hostInput = (unsigned int*) malloc(inputLength * sizeof(unsigned int));
    hostBins = (unsigned int*) malloc(NUM_BINS * sizeof(unsigned int));
    resultRef = (unsigned int*) malloc(NUM_BINS * sizeof(unsigned int));

    //@@ Insert code below to initialize hostInput to random numbers whose values range from 0 to (NUM_BINS - 1)
    //@@ Insert code below to create reference result in CPU
    // rand() % (max_number + 1 - minimum_number) + minimum_number
    memset(resultRef, 0, NUM_BINS*sizeof(*resultRef)); // Set array elements to zero
    for (int i = 0; i < inputLength; ++i) {
        unsigned int randomNumber = rand() % NUM_BINS;
        hostInput[i] = randomNumber;
        if (resultRef[randomNumber] < 127) {
            resultRef[randomNumber] += 1;
        }
    }

    //@@ Insert code below to allocate GPU memory here
    cudaMalloc(&deviceInput, inputLength * sizeof(unsigned int));
    cudaMalloc(&deviceBins, NUM_BINS * sizeof(unsigned int));

    //@@ Insert code to Copy memory to the GPU here
    cudaMemcpy(deviceInput, hostInput, inputLength * sizeof(unsigned int), cudaMemcpyHostToDevice);

    //@@ Insert code to initialize GPU results
    cudaMemset(deviceBins, 0, NUM_BINS * sizeof(unsigned int));

    //@@ Initialize the grid and block dimensions here
    int Db_h = 64;
    int Dg_h = (inputLength + Db_h - 1) / Db_h;

    //@@ Launch the GPU Kernel here
    histogram_kernel<<<Dg_h, Db_h>>>(deviceInput, deviceBins, inputLength, NUM_BINS);

    //@@ Initialize the second grid and block dimensions here
    int Db_c = 64;
    int Dg_c = (NUM_BINS + Db_c - 1) / Db_c;

    //@@ Launch the second GPU Kernel here
    convert_kernel<<<Dg_c, Db_c>>>(deviceBins, NUM_BINS);

    //@@ Copy the GPU memory back to the CPU here
    cudaMemcpy(hostBins, deviceBins, NUM_BINS * sizeof(unsigned int), cudaMemcpyDeviceToHost);

    //@@ Insert code below to compare the output with the reference
    int equality = 1;
    for (int i = 0; i < inputLength; ++i) {
        //printf("CPU element: %f\n", resultRef[i]);
        //printf("GPU element: %f\n", hostBins[i]);
        if (hostBins[i] != resultRef[i]) {
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
    cudaFree(deviceInput);
    cudaFree(deviceBins);

    //@@ Free the CPU memory here
    free(hostInput);
    free(hostBins);
    free(resultRef);

    return 0;
}