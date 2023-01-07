#include <stdio.h>
#include <sys/time.h>

//#define DataType double
#define DataType float
#define DEBUG 0

// Compute C = A * B (on CPU)
void matmul(DataType *A, DataType *B, DataType *C, int numARows,
            int numAColumns, int numBRows, int numBColumns) {
    // Multiplying matrices A and B and storing it in C
    for (int i = 0; i < numARows; ++i) {
        for (int j = 0; j < numBColumns; ++j) {
            C[i*numBColumns + j] = 0.0;
            for (int k = 0; k < numAColumns; ++k) {
                C[i*numBColumns + j] += A[i*numAColumns + k] * B[k*numBColumns + j];
            }
            #if DEBUG
            printf("C[%d, %d] = %f\n", i, j, C[i*numBColumns + j]);
            #endif
        }
    }
}

// Compute C = A * B (on GPU)
__global__ void gemm(DataType *A, DataType *B, DataType *C, int numARows,
                      int numAColumns, int numBRows, int numBColumns){
    //@@ Insert code to implement matrix multiplication here
    // x corresponds to the number of columns, y corresponds to the number of rows
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    if ((col >= numBColumns) || (row >= numARows)) return;

    DataType tmpSum = 0.0;
    for (int k = 0; k < numAColumns; ++k) {
        tmpSum += A[row*numAColumns + k] * B[k*numBColumns + col];
    }
    C[row*numBColumns + col] = tmpSum;
    #if DEBUG
    printf("C[%d, %d] = %f\n", row, col, C[row*numBColumns + col]);
    #endif
}

int main(int argc, char **argv) {
    DataType *umA; // The A matrix
    DataType *umB; // The B matrix
    DataType *umC; // The output C matrix
    DataType *resultRef; // The reference result
    int numARows;    // number of rows in the matrix A
    int numAColumns; // number of columns in the matrix A
    int numBRows;    // number of rows in the matrix B
    int numBColumns; // number of columns in the matrix B
    int numCRows;
    int numCColumns;

    //@@ Insert code below to read in numARows, numAColumns, numBColumns from args
    if (argc != 5) {
        printf("ERROR: Exactly four input parameters are required to run the program (%d != 4).\n", argc);
        exit(1);
    }
    numARows = atoi(argv[1]);
    numAColumns = atoi(argv[2]);
    numBRows = atoi(argv[3]);
    numBColumns = atoi(argv[4]);
    numCRows = numARows;
    numCColumns = numBColumns;
    printf("Input matrix dim (%d x %d) (%d x %d) (%d x %d)\n", numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns);
    if (numAColumns != numBRows) {
        printf("ERROR: Matrix A must have the same number of columns as the number of rows of matrix B (%d != %d).\n", numAColumns, numBRows);
        return 0;
    }

    //@@ Insert code below to allocate unified memory memory for input and output
    cudaMallocManaged(&umA, numARows * numAColumns * sizeof(DataType)); // umA[i*numAColumns + j] is equivalent to the usual umA[i][j]
    cudaMallocManaged(&umB, numBRows * numBColumns * sizeof(DataType)); // umB[i*numBColumns + j] is equivalent to the usual umB[i][j]
    cudaMallocManaged(&umC, numCRows * numCColumns * sizeof(DataType)); // umC[i*numCColumns + j] is equivalent to the usual umC[i][j]
    cudaMallocManaged(&resultRef, numCRows * numCColumns * sizeof(DataType));

    //@@ Insert code below to initialize umA and umB to random numbers, and create reference result in CPU
    for (int i = 0; i < numARows; ++i) {
        for (int j = 0; j < numAColumns; ++j) {
            DataType randomNumber = rand() / (DataType) RAND_MAX; // Random number in interval [0, 1.0]
            umA[i*numAColumns + j] = randomNumber;
            #if DEBUG
            printf("umA[%d, %d] = %f\n", i, j, umA[i*numBColumns + j]);
            #endif
        }
    }
    for (int i = 0; i < numBRows; ++i) {
        for (int j = 0; j < numBColumns; ++j) {
            DataType randomNumber = rand() / (DataType) RAND_MAX; // Random number in interval [0, 1.0]
            umB[i*numBColumns + j] = randomNumber;
            #if DEBUG
            printf("umB[%d, %d] = %f\n", i, j, umB[i*numBColumns + j]);
            #endif
        }
    }
    // Calculate reference result
    matmul(umA, umB, resultRef, numARows, numAColumns, numBRows, numBColumns);

    //@@ Initialize the grid and block dimensions here
    // x corresponds to the number of columns, y corresponds to the number of rows
    int Dbx = 16;
    int Dby = 16;
    int Dgx = (numCColumns + Dbx - 1) / Dbx;
    int Dgy = (numCRows + Dby - 1) / Dby;

    //@@ Launch the GPU Kernel here
    gemm <<<dim3(Dgx, Dgy, 1), dim3(Dbx, Dby, 1)>>>(umA, umB, umC, numARows, numAColumns, numBRows, numBColumns);

    // Explicitly synchronize on the host side before directly accessing the output of the kernel
    cudaDeviceSynchronize();

    //@@ Insert code below to compare the output with the reference
    int equality = 1;
    for (int i = 0; i < numCRows; ++i) {
        for (int j = 0; j < numCColumns; ++j) {
            if (fabs(umC[i*numCColumns + j] - resultRef[i*numCColumns + j]) > 1e-4) { // Compare if elements are approximately equal
                equality = 0;
                #if DEBUG
                printf("Position: [%d, %d], Difference: %f\n", i, j, fabs(umC[i*numCColumns + j] - resultRef[i*numCColumns + j]));
                #endif
                break;
            }
        }
    }
    if (equality == 1) {
        printf("CPU and GPU results are equal.\n");
    } else {
        printf("CPU and GPU results are NOT equal.\n");
    }

    //@@ Free the unified memory here
    cudaFree(umA);
    cudaFree(umB);
    cudaFree(umC);
    cudaFree(resultRef);

    return 0;
}