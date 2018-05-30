#include "gpu_func.h"
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <iostream>
#include "cublas_v2.h"
#include <cmath>

#define BLOCK_SIZE 16


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

/*
Routine to perform an in-place GEMM operation, i.e., C := alpha*A*B + beta*C
*/

__global__
void myGEMMkernel(double* A, double* B, double* C, double alpha, double beta, int M,
           int N, int K,bool AT,bool BT) 
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    double inner_prod=0.0;
    if(row<M&&col<N){
        for(int k = 0; k<K; k++){
            int indexA;
            int indexB;
            if (AT){
                 indexA = (row*K)+k;
                 indexB = (col*K)+k;
            
            }

            else if(BT){
                indexA = (k*M)+row;
                indexB = (k*N)+col;
            
            }

            else{
                indexA = (k*M)+row;
                indexB = (col*K)+k;
            
            }
            inner_prod+=A[indexA]*B[indexB];
        }
        C[col*M+row] =alpha*inner_prod+beta*C[col*M+row];
    }
}

int myGEMM(double* A, double* B, double* C, double* alpha, double* beta, int M,
           int N, int K,bool AT,bool BT) 
{
    /* TODO: Write an efficient GEMM implementation on GPU */
    dim3 dimBlock(32,32);
    dim3 dimGrid((N+dimBlock.x-1)/dimBlock.x, (M+dimBlock.y-1)/dimBlock.y);

    myGEMMkernel<<<dimGrid, dimBlock>>>(A,B,C,*alpha,*beta,M,N,K,AT,BT);

    return 0;
}

__global__
void softmax_kernel(const double* Z, double* A,int M, int N)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    double denom = 0.0;
    for (int i =0;i<M;i++)
    {
        denom+=exp(Z[col*M+i]);
    }
    for (int i =0;i<M;i++)
    {
        A[col*M+i]=exp(Z[col*M+i])/denom;
    }
}

void softmax_p(const double* Z, double* A,int M, int N)
{
    dim3 dimBlock(32);
    dim3 dimGrid((N+dimBlock.y-1)/dimBlock.x);
    softmax_kernel<<<dimGrid,dimBlock>>>(Z,A,M,N);

}

__global__
void sigmoid_kernel(const double* Z, double* A,int M,int N)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if(row<M&&col<N)
    {
        A[M*col+row]=1.0/(1.0+exp(-1*Z[M*col+row]));
    }
}

void sigmoid_p(const double* Z, double* A,int M, int N)
{
    dim3 dimBlock(32,32);
    dim3 dimGrid((N+dimBlock.x-1)/dimBlock.x, (M+dimBlock.y-1)/dimBlock.y);
    sigmoid_kernel<<<dimGrid,dimBlock>>>(Z,A,M,N);
}

__global__
void row_sum_kernel(double* W, double* Y, int M, int N)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if(row<M)
    {
        double sum=0.0;
        for(int i=0;i<N;i++){
            sum+=W[i*M+row];
        }
        Y[row]=sum;
    }
    
}


void row_sum(double* W, double* Y, int M, int N)
{
    dim3 dimBlock(32);
    dim3 dimGrid((N+dimBlock.x-1)/dimBlock.x);
    row_sum_kernel<<<dimGrid,dimBlock>>>(W,Y,M,N);


}
__global__
void elem_mult_kernel(double* A, double* B, double* C,int M,int N)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if(row<M&&col<N)
    {
        C[(col*M+row)]=A[(col*M+row)]*B[(col*M+row)];
    }

}
void elem_mult(double* A, double* B, double* C,int M,int N)
{
    dim3 dimBlock(32,32);
    dim3 dimGrid((N+dimBlock.x-1)/dimBlock.x, (M+dimBlock.y-1)/dimBlock.y);
    elem_mult_kernel<<<dimGrid,dimBlock>>>(A,B,C,M,N);
}
