#include <stdio.h>
#include <sys/time.h>
#include <random>

#define NUM_BINS 4096
#define MAX_VAL 127
#define DEBUG 0
#define SHARED_MEM 0

void histogram_cpu(unsigned int *input, unsigned int *bins,
                   unsigned int num_elements, unsigned int num_bins) {
    // Set array elements to zero
    memset(bins, 0, num_bins * sizeof(*bins));

    // Count array elements
    for (int i = 0; i < num_elements; ++i) {
        unsigned int num = input[i];
        if (bins[num] < MAX_VAL) {
            bins[num] += 1;
        }
    }
}

__global__ void histogram_kernel(unsigned int *input, unsigned int *bins,
                                 unsigned int num_elements, unsigned int num_bins) {
    //@@ Insert code below to compute histogram of input using shared memory and atomics
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    #if SHARED_MEM
    // Each thread block creates its own shared histogram, stores in a shared variable, and copies the result to the global variable
    // Shared variable
    __shared__ unsigned int s_bins[NUM_BINS];

    // Initaliaze shared histogram to zero
    if (blockDim.x < num_bins) { // each thread must potentially set more than one location to zero
        for (int i = threadIdx.x; i < num_bins; i += blockDim.x) {
            if (i < num_bins) {
                s_bins[i] = 0;
            }
        }
    } else { // each thread must set max. one location to zero
        if (threadIdx.x < num_bins) {
            s_bins[threadIdx.x] = 0;
        }
    }
    // synchronize all threads of the block
    __syncthreads();


    // update shared histogram of this thread block
    if (idx < num_elements) {
        atomicAdd(&(s_bins[input[idx]]), 1);
    }
    // synchronize all threads of the block
    __syncthreads();


    // Add entries of shared histogram to global histogram
    if (blockDim.x < num_bins) { // each thread must potentially copy more than one location
        for (int i = threadIdx.x; i < num_bins; i += blockDim.x) {
            if (i < num_bins) {
                atomicAdd(&(bins[i]), s_bins[i]);
            }
        }
    } else { // each thread must only copy max. one location
        if (threadIdx.x < num_bins) {
            atomicAdd(&(bins[threadIdx.x]), s_bins[threadIdx.x]);
        }
    }

    #else
    // Directly write to global memory
    if (idx >= num_elements) return;
    atomicAdd(&bins[input[idx]], 1);
    #endif
}

__global__ void convert_kernel(unsigned int *bins, unsigned int num_bins) {
    //@@ Insert code below to clean up bins that saturate at MAX_VAL
    const int bin = blockIdx.x * blockDim.x + threadIdx.x;
    if (bin >= num_bins) return;

    if (bins[bin] > MAX_VAL) {
        bins[bin] = MAX_VAL;
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
    for (int i = 0; i < inputLength; ++i) {
        hostInput[i] = rand() % NUM_BINS; // Formula: rand() % (max_number + 1 - minimum_number) + minimum_number
        #if DEBUG
        printf("hostInput[%d] =  %d\n", i, hostInput[i]);
        #endif
    }

    //@@ Insert code below to create reference result in CPU
    histogram_cpu(hostInput, resultRef, inputLength, NUM_BINS);

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
    for (int i = 0; i < NUM_BINS; ++i) {
        #if DEBUG
        printf("resultRef[%d] = %d\n", i, resultRef[i]);
        printf("hostBins[%d] =  %d\n", i, hostBins[i]);
        #endif
        if (hostBins[i] != resultRef[i]) {
            equality = 0;
            //break;
        }
    }
    if (equality == 1) {
        printf("CPU and GPU results are equal.\n");
    } else {
        printf("CPU and GPU results are NOT equal.\n");
    }

    // Write histogram to file
    FILE *fptr;
    fptr = fopen("./histogram.txt","w+");
    if (fptr == NULL) {
        printf("Error!");   
        exit(1);             
    }
    for (int i = 0; i < NUM_BINS; ++i) {
        fprintf(fptr, "%d\n", hostBins[i]);
    }
    fclose(fptr);

   FILE *fp;

   fp = fopen("./test.txt", "w+");
   fprintf(fp, "This is testing for fprintf...\n");
   fprintf(fp, "This is testing for fprintf... %d \n", 10);
   fputs("This is testing for fputs...\n", fp);
   fclose(fp);

    //@@ Free the GPU memory here
    cudaFree(deviceInput);
    cudaFree(deviceBins);

    //@@ Free the CPU memory here
    free(hostInput);
    free(hostBins);
    free(resultRef);

    return 0;
}