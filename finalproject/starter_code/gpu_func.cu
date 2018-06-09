#include "gpu_func.h"
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <iostream>
#include "cublas_v2.h"
#include <cmath>

#define BLOCK_SIZE 16
#define BLOCK_X 32
#define BLOCK_Y 32
/*
typedef struct {
     int width;
     int height;
     int stride;
     double* elements;
     //Matrix(){}
     //Matrix(double* A, int M, int N): width(N), height(M), elements(A),
     //                                 stride(N){}
} Matrix; 

// Matrices are stored in row-major order:
// M(row, col) = *(M.elements + row * M.stride + col)

// Get a matrix element
__device__ float GetElement(const double*  A, int row, int col,int heigth)
{
     return A[col*heigth + row];
}

// Set a matrix element
__device__ void SetElement(double* A, int row, int col, int heigth,
         double value)
{
     A[col*heigth + row] = value;
}

// Get the BLOCK_SIZExBLOCK_SIZE sub-matrix Asub of A that is
// located col sub-matrices to the right and row sub-matrices down
// from the upper-left corner of A
__device__ Matrix GetSubMatrix(Matrix A, int row, int col)
{
     Matrix Asub;
     Asub.width = BLOCK_SIZE;
     Asub.height = BLOCK_SIZE;
     Asub.stride = A.stride;
     Asub.elements = &A.elements[A.stride * BLOCK_SIZE * row
        + BLOCK_SIZE * col];
     return Asub;
}

// Forward declaration of the matrix multiplication kernel
__global__ void MatMulKernel(const Matrix, const Matrix, Matrix);
// Matrix multiplication - Host code
// Matrix dimensions are assumed to be multiples of BLOCK_SIZE

// Matrix multiplication kernel called by MatMul()
__global__ 
void MatMulKernel(double* A, double* B, double* C,int M, int N, int K)
{
     // Block row and column
     int blockRow = blockIdx.y;
     int blockCol = blockIdx.x;
     // Each thread block computes one sub-matrix Csub of C
     double* Csub = &C[BLOCK_SIZE * blockRow
        + BLOCK_SIZE * blockCol * M];
     //Matrix Csub = GetSubMatrix(C, blockRow, blockCol);
     
     // Each thread computes one element of Csub
     // by accumulating results into Cvalue
     double Cvalue = 0;

     // Thread row and column within Csub
     int row = threadIdx.y;
     int col = threadIdx.x;
      // Loop over all the sub-matrices of A and B that are
      // required to compute Csub
      // Multiply each pair of sub-matrices together
      // and accumulate the results
      for (int m = 0; m < (K / BLOCK_SIZE); ++m) {
           // Get sub-matrix Asub of A
     double* Asub = &A[M * BLOCK_SIZE * blockRow
        + BLOCK_SIZE * m];
           //Matrix Asub = GetSubMatrix(A, blockRow, m);
           // Get sub-matrix Bsub of B
     double* Bsub = &B[M * BLOCK_SIZE * m
        + BLOCK_SIZE * blockCol];
           //Matrix Bsub = GetSubMatrix(B, m, blockCol);
            // Shared memory used to store Asub and Bsub respectively
            __shared__ double As[BLOCK_SIZE][BLOCK_SIZE];
            __shared__ double Bs[BLOCK_SIZE][BLOCK_SIZE];
            // Load Asub and Bsub from device memory to shared memory
            // Each thread loads one element of each sub-matrix
            As[row][col] = GetElement(Asub, row, col, M);
            Bs[row][col] = GetElement(Bsub, row, col, K);
            // Synchronize to make sure the sub-matrices are loaded
            // before starting the computation
            __syncthreads();
            // Multiply Asub and Bsub together
            for (int e = 0; e < BLOCK_SIZE; ++e)
                 Cvalue += As[row][e] * Bs[e][col];
             // Synchronize to make sure that the preceding
             // computation is done before loading two new
             // sub-matrices of A and B in the next iteration
             __syncthreads();
      }
       // Write Csub to device memory
       // Each thread writes one element
      SetElement(Csub, row, col, Cvalue, M);
}
void MatMul(const double* A, const double* B, double* C)
{

     // Invoke kernel
     dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
     dim3 dimGrid(B.width / dimBlock.x, A.height / dimBlock.y);
     //MatMulKernel<<<dimGrid, dimBlock>>>(A, B, C);
          // Read C from device memory
}
*/
__global__
void matrixMulCUDA(float* C, float* A, float* B, int wA, int wB) {
    // Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Index of the first sub-matrix of A processed by the block
    int aBegin = wA * BLOCK_SIZE * by;

    // Index of the last sub-matrix of A processed by the block
    int aEnd   = aBegin + wA - 1;

    // Step size used to iterate through the sub-matrices of A
    int aStep  = BLOCK_SIZE;

    // Index of the first sub-matrix of B processed by the block
    int bBegin = BLOCK_SIZE * bx;

    // Step size used to iterate through the sub-matrices of B
    int bStep  = BLOCK_SIZE * wB;

    // Csub is used to store the element of the block sub-matrix
    // that is computed by the thread
    float Csub = 0;

    // Loop over all the sub-matrices of A and B
    // required to compute the block sub-matrix
    for(int a = aBegin, b = bBegin;
            a <= aEnd;
            a += aStep, b += bStep) {

        // Declaration of the shared memory array As used to
        // store the sub-matrix of A
        __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];

        // Declaration of the shared memory array Bs used to
        // store the sub-matrix of B
        __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

        // Load the matrices from device memory
        // to shared memory; each thread loads
        // one element of each matrix
        As[ty][tx] = A[a + wA * ty + tx];
        Bs[ty][tx] = B[b + wB * ty + tx];

        // Synchronize to make sure the matrices are loaded
        __syncthreads();

        // Multiply the two matrices together;
        // each thread computes one element
        // of the block sub-matrix
#pragma unroll

        for(int k = 0; k < BLOCK_SIZE; ++k) {
            Csub += As[ty][k] * Bs[k][tx];
        }

        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
    }

    // Write the block sub-matrix to device memory;
    // each thread writes one element
    int c = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
    C[c + wB * ty + tx] = Csub;
}

__global__
void GEMM_shared(double* A, double* B,double*C, double alpha, double beta, int M, int N,
        int K)
{
   const unsigned int bx = BLOCK_X, by = BLOCK_Y;
   const unsigned int tx = threadIdx.x, ty = threadIdx.y;
   const unsigned int I = blockIdx.x*bx + tx, J = blockIdx.y*by + ty;
   const unsigned int gx = gridDim.x, gy = gridDim.y;
   __shared__ double Asub[BLOCK_X][BLOCK_Y];
   __shared__ double Bsub[BLOCK_X][BLOCK_Y];

   if(I<M && J<N)
   {
       double c = 0.0;
       for (unsigned int k=0; k < gy; k++){
           Asub[tx][ty] = A[ J*M+k*by+ty];
           Bsub[ty][tx] = B[J+N*(k*bx+tx)];
            __syncthreads(); // Synchronizes all threads in a block
          for (unsigned int kk=0; kk< bx; kk++){
               c +=Asub[kk][tx]*Bsub[kk][ty];
          }
           __syncthreads(); // Avoids memory hazards
          }
     C[J*M+I] = alpha*c+beta*C[J*M+I];
   }

}
__global__
void device_add_one(int* d_result, int t) {
    *d_result = t + 1;
}

/*
Just a dummy function that can be used to warm up GPU
*/
int useless_gpu_add_one(int t) {
    int result;
    int* d_result;

    checkCudaErrors(cudaMalloc((void**)&d_result, 1 * sizeof(int)));

    event_pair timer;
    start_timer(&timer);
    device_add_one<<<1,1>>>(d_result, t);
    check_launch("device_add_one");
    double time = stop_timer(&timer);

    std::cout << "device_add_one took: " << time << " seconds" << std::endl;

    checkCudaErrors(cudaMemcpy(&result, d_result, 1 * sizeof(int),
                               cudaMemcpyDeviceToHost));
    return result;
}

__global__
void myGEMMkernel(double* A, double* B, double* C, double alpha, double beta, int M,
           int N, int K,bool AT,bool BT) 
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    
    if(row<M && col<N )
    {
        double inner_prod=0.0;
        int a_ind;
        int b_ind;
        for(int i = 0; i < K; i++) {
            a_ind = row + (i*M);
            b_ind = i + (col * K);
            inner_prod += A[a_ind] * B[b_ind];
        }
    
        C[col*M+row] =alpha*inner_prod+beta*C[col*M+row];
    }
}

__global__
void myGEMMkernel1(double* A, double* B, double* C, double alpha, double beta, int M,
           int N, int K) 
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if(row<M && col< N)
    {
        int ic=(col*M)+row;
        double inner_prod=0.0;
        int ia;
        int ib;
        for(int i=0;i<K;i++)
        {
            ia=(row*K)+i;
            ib=i+(col*K);
            inner_prod+= A[ia]*B[ib];
        }
        C[ic] =alpha*inner_prod+beta*C[ic];
        
    }

}

__global__
void myGEMMkernel2(double* A, double* B, double* C, double alpha, double beta, int M,
           int N, int K) 
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if(row<M && col< N)
    {
        int ic=(col*M)+row;
        double inner_prod=0.0;
        int ia;
        int ib;
        for(int i=0;i<K;i++)
        {
            ia=(i*M)+row;
            ib=col+(i*N);
            inner_prod+= A[ia]*B[ib];
        }
        C[ic] =alpha*inner_prod+beta*C[ic];
        
    }

}
int myGEMM(double* A, double* B, double* C, double* alpha, double* beta, int M,
           int N, int K,bool AT,bool BT) 
{


   
    if(AT){
        dim3 dimBlock(32,6);
        dim3 dimGrid((M+dimBlock.x-1)/dimBlock.x, (N+dimBlock.y-1)/dimBlock.y);
        myGEMMkernel1<<<dimGrid, dimBlock>>>(A,B,C,*alpha,*beta,M,N,K);
    }
   
    else if(BT){
        dim3 dimBlock(32,6);
        dim3 dimGrid((M+dimBlock.x-1)/dimBlock.x, (N+dimBlock.y-1)/dimBlock.y);
        myGEMMkernel2<<<dimGrid, dimBlock>>>(A,B,C,*alpha,*beta,M,N,K);
    }
    
    else {
        dim3 dimBlock(BLOCK_X,BLOCK_Y);
        dim3 dimGrid((M+dimBlock.x-1)/dimBlock.x, (N+dimBlock.y-1)/dimBlock.y);
        GEMM_shared<<<dimGrid, dimBlock>>>(A,B,C,*alpha,*beta,M,N,K);
    }
        return 0;
}

/* GPU kernel for 10-class softmax */
__global__
void gpuSoftmax_kernel(double* A, unsigned int num_classes, unsigned int N) {
    int col = blockIdx.x*blockDim.x + threadIdx.x;

    if (col < N) {
        double denominator = 0.0;

        for(int c = 0; c < num_classes; c++){
            denominator += (double) std::exp(A[col*num_classes + c]);
        }

        for(int c = 0; c < num_classes; c++){
            int ij = c + (col * num_classes);
            A[ij] = (double) std::exp(A[ij])/ (double) denominator;
        }
    }
}

__global__
void softmax_kernel(double* A,int M, int N)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    double denom = 0.0;
    if(col<N){
    for (int i =0;i<M;i++)
    {
        denom+=(double) std::exp(A[col*M+i]);
    }
    for (int i =0;i<M;i++)
    {
        A[col*M+i]=std::exp(A[col*M+i])/(double)denom;
    }
    }
}

__global__
void sigmoid_kernel(double* A,int M,int N)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if(row<M&&col<N)
    {
        int ind = row + (col * M);
        A[ind] = (double) 1.0 / (double)(1.0 + exp(-1.0 * A[ind]));
    }
}

/* Routine for in-place element-wise sigmoid */
void gpuSigmoid(double* A, unsigned int num_neurons, unsigned int N) {
    unsigned int num_threads = 192;
    unsigned int thr_x = 32;
    unsigned int thr_y = (num_threads + thr_x - 1) / thr_x;
    dim3 threads(thr_x, thr_y);

    unsigned int blk_x = (num_neurons + thr_x - 1) / thr_x;
    unsigned int blk_y = (N + thr_y - 1) / thr_y;
    dim3 blocks(blk_x, blk_y);

    sigmoid_kernel<<< blocks, threads >>>(A, num_neurons, N);
    check_launch("gpuSigmoid_kernel");
}


__global__
void row_sum_kernel(double* W, double* Y, int M, int N)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if(row<M)
    {
        double sum=0.0;
        for(int i=0;i<N;i++){
            sum+=W[(i*M)+row];
        }
        Y[row]=sum;
    }
    
}

/* GPU kernel for broadcasting sum for matrix A with vector v */
__global__
void gpuMatVecSum_kernel(double *A, double *v, int M, int N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x; 
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < M &&  col < N) {
        int ind = row + (M*col);
        double num = v[row];
        A[ind] += num;
    }
}

/* Routine for broadcasting sum for matrix A with vector v */
void gpuMatVecSum(double *A, double *v, int M, int N) {
    unsigned int num_threads = 192;
    unsigned int thr_x = 32;
    unsigned int thr_y = (num_threads + thr_x - 1) / thr_x;
    dim3 threads(thr_x, thr_y);

    unsigned int blk_x = (M + thr_x - 1) / thr_x;
    unsigned int blk_y = (N + thr_y - 1) / thr_y;
    dim3 blocks(blk_x, blk_y);
    gpuMatVecSum_kernel<<< blocks, threads >>>(A, v, M, N);
    check_launch("gpuMatVecSum_kernel");
}

/* GPU kernel for elementwise matrix sum */

__global__
void elem_add_kernel(double* A, double* B, double* C,double alpha, double beta,int M,int N)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if(row<M&&col<N)
    {
        int index =(M*col)+row;
        C[index]=(alpha*A[index])+(beta*B[index]);
    }

}


/* GPU kernel for derivative of sigmoid */
/** Routine for derivative of sigmoid */

__global__
void sigmoid_back_kernel(double *A, double *B, double *C, int M, int N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < M && col < N) {
        int index =(M*col)+row;
        C[index]=(double)A[index] * B[index] * (1.0 -B[index]);
    }

}


void softmax_p(double* A,int M, int N)
{
    dim3 dimBlock(32);
    dim3 dimGrid((N+dimBlock.x-1)/dimBlock.x);
    softmax_kernel<<<dimGrid,dimBlock>>>(A,M,N);

}


void sigmoid_p(double* A,int M, int N)
{
    dim3 dimBlock(32,6);
    dim3 dimGrid((M+dimBlock.x-1)/dimBlock.x, (N+dimBlock.y-1)/dimBlock.y);
    sigmoid_kernel<<<dimGrid,dimBlock>>>(A,M,N);
}



void row_sum(double* W, double* Y, int M, int N)
{
    dim3 dimBlock(192);
    dim3 dimGrid((M+dimBlock.x-1)/dimBlock.x);
    row_sum_kernel<<<dimGrid,dimBlock>>>(W,Y,M,N);


}


void elem_add(double* A, double* B, double* C,double alpha, double beta,int M,int N)
{
    dim3 dimBlock(32,32);
    dim3 dimGrid((M+dimBlock.x-1)/dimBlock.x, (N+dimBlock.y-1)/dimBlock.y);
    elem_add_kernel<<<dimGrid,dimBlock>>>(A,B,C,alpha,beta,M,N);
}

void sigmoid_back(double *A, double *B, double *C, int M, int N) {
    dim3 dimBlock(32,6);
    dim3 dimGrid((M+dimBlock.x-1)/dimBlock.x, (N+dimBlock.y-1)/dimBlock.y);
    sigmoid_back_kernel<<< dimGrid, dimBlock >>>(A, B, C, M, N);
}

