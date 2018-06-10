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

__global__
void shared_GEMM_kernel(double* A, double* B,double*C, double alpha,double beta, int M, int N,int K ) {
    // Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = bx*BLOCK_SIZE + tx;
    int col = by*BLOCK_SIZE + ty;

    double Csub = 0.0;
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE+1];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE+1];

    // Loop over all the sub-matrices of A and B
    // required to compute the block sub-matrix
    int lim = (K+BLOCK_SIZE -1 )/BLOCK_SIZE;
    for(int k=0;
            k < lim;
            k++) {

        // Declaration of the shared memory array As used to
        // store the sub-matrix of A

        // Declaration of the shared memory array Bs used to
        // store the sub-matrix of B

        // Load the matrices from device memory
        // to shared memory; each thread loads
        // one element of each matrix
        //As[tx][ty]=0.0;
        
        if((BLOCK_SIZE*k)+ty<K)
        {
            double * Aview =A+(k*BLOCK_SIZE*M+BLOCK_SIZE*bx);
            As[tx][ty]=Aview[(ty*M)+tx];
        }
        else{
            As[tx][ty]=0.0;
        }

        if((BLOCK_SIZE*k+tx)<K)
        {
            double * Bview =B+(by*BLOCK_SIZE*K+BLOCK_SIZE*k);
            Bs[tx][ty]=Bview[(ty*K)+tx];
        }
        else{
            Bs[tx][ty]=0.0;
        }

        // Synchronize to make sure the matrices are loaded
        __syncthreads();

        // Multiply the two matrices together;
        // each thread computes one element
        // of the block sub-matrix
#pragma unroll

        for(int k = 0; k < BLOCK_SIZE; ++k) {
            Csub += alpha*As[ty][k] * Bs[k][tx];
        }

        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
    }

    // Write the block sub-matrix to device memory;
    // each thread writes one element
    
    if((BLOCK_SIZE*bx+tx)<M &&(BLOCK_SIZE*by+ty) <N)
    {
        Csub+=beta*C[col*M+row];
        double* Cview = C+(BLOCK_SIZE*by*M+BLOCK_SIZE*bx);
        Cview[ty*M+tx] = Csub;
    }
    
}

__global__
void shared_GEMM_kernel1(double* A, double* B,double*C, double alpha,double beta, int M, int N,int K ) {
    // Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = bx*BLOCK_SIZE + tx;
    int col = by*BLOCK_SIZE + ty;

    double Csub = 0.0;
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    // Loop over all the sub-matrices of A and B
    // required to compute the block sub-matrix
    int lim = (K+BLOCK_SIZE -1 )/BLOCK_SIZE;
    for(int k=0;
            k < lim;
            k++) {

        // Declaration of the shared memory array As used to
        // store the sub-matrix of A

        // Declaration of the shared memory array Bs used to
        // store the sub-matrix of B

        // Load the matrices from device memory
        // to shared memory; each thread loads
        // one element of each matrix
        //As[tx][ty]=0.0;
        
        if((BLOCK_SIZE*k)<(K-ty))
        {
            double * Aview =A+(bx*BLOCK_SIZE*M+BLOCK_SIZE*k);
            As[tx][ty]=Aview[(ty*M)+tx];
        }
        else{
            As[tx][ty]=0.0;
        }
        

        if((BLOCK_SIZE*k)<(K-tx))
        {
            double * Bview =B+(by*BLOCK_SIZE*K+BLOCK_SIZE*k);
            Bs[tx][ty]=Bview[(ty*K)+tx];
        }
        else{
            Bs[tx][ty]=0.0;
        }

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
    
    if(row<M && col<N)
    {
        C[col*M+row] =alpha*Csub+beta*C[col*M+row];
    }
    
}
__global__
void shared_GEMM_kernel2(double* A, double* B,double*C, double alpha,double beta, int M, int N,int K ) {
    // Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = bx*BLOCK_SIZE + tx;
    int col = by*BLOCK_SIZE + ty;

    double Csub = 0.0;
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    // Loop over all the sub-matrices of A and B
    // required to compute the block sub-matrix
    int lim = (K+BLOCK_SIZE -1 )/BLOCK_SIZE;
    for(int k=0;
            k < lim;
            k++) {

        // Declaration of the shared memory array As used to
        // store the sub-matrix of A

        // Declaration of the shared memory array Bs used to
        // store the sub-matrix of B

        // Load the matrices from device memory
        // to shared memory; each thread loads
        // one element of each matrix
        //As[tx][ty]=0.0;
        
        if((BLOCK_SIZE*k)<(K-ty))
        {
            double * Aview =A+(k*BLOCK_SIZE*M+BLOCK_SIZE*bx);
            As[tx][ty]=Aview[(ty*M)+tx];
        }
        else{
            As[tx][ty]=0.0;
        }
        

        if((BLOCK_SIZE*k)<(K-tx))
        {
            double * Bview =B+(k*BLOCK_SIZE*N+BLOCK_SIZE*by);
            Bs[tx][ty]=Bview[(ty*N)+tx];
        }
        else{
            Bs[tx][ty]=0.0;
        }

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
    
    if(row<M && col<N)
    {
        C[col*M+row] =alpha*Csub+beta*C[col*M+row];
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
           int N, int K) 
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
        dim3 dimBlock(BLOCK_SIZE,BLOCK_SIZE);
        dim3 dimGrid((M+dimBlock.x-1)/dimBlock.x, (N+dimBlock.y-1)/dimBlock.y);
        shared_GEMM_kernel2<<<dimGrid, dimBlock>>>(A,B,C,*alpha,*beta,M,N,K);
    }
   
    else if(BT){
        dim3 dimBlock(BLOCK_SIZE,BLOCK_SIZE);
        dim3 dimGrid((M+dimBlock.x-1)/dimBlock.x, (N+dimBlock.y-1)/dimBlock.y);
        shared_GEMM_kernel2<<<dimGrid, dimBlock>>>(A,B,C,*alpha,*beta,M,N,K);
    }
    
    else {
        dim3 dimBlock(BLOCK_SIZE,BLOCK_SIZE);

        dim3 dimGrid((M+dimBlock.x-1)/dimBlock.x, (N+dimBlock.y-1)/dimBlock.y);

        cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
        
        shared_GEMM_kernel<<<dimGrid, dimBlock>>>(A,B,C,*alpha,*beta,M,N,K);
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

