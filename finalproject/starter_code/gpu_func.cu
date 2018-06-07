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

_global__
void myGEMM_kernel(double* A, double* B, double* C,
                   double alpha, double beta,
                   int M, int N, int K,
                   bool AT, bool BT) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < M && col < N) {
        int c_ind = row + (col * M);
        double dot_prod = 0.0;
        int a_ind;
        int b_ind;
        for(int i = 0; i < K; i++) {
            if (AT)
                a_ind = (row*K) + i;
            else
                a_ind = row + (i*M);
            if (BT)
                b_ind = col + (i*N);
            else
                b_ind = i + (col * K);
            dot_prod += A[a_ind] * B[b_ind];
        }
        C[c_ind] = (alpha * dot_prod) + (beta * C[c_ind]);
    }
}

/*
Routine to perform an in-place GEMM operation, i.e., C := alpha*A*B + beta*C
*/
int myGEMM(double* A, double* B, double* C,
           double* alpha, double* beta,
           int M, int N, int K,
           bool AT, bool BT) {
    /* TODO: Write an efficient GEMM implementation on GPU */
    unsigned int num_threads = 192;
    unsigned int thr_x = 32;
    unsigned int thr_y = (num_threads + thr_x - 1) / thr_x;
    
    dim3 threads(thr_x, thr_y);

    unsigned int blk_x = (M + thr_x - 1) / thr_x;
    unsigned int blk_y = (N + thr_y - 1) / thr_y;
    dim3 blocks(blk_x, blk_y);

    myGEMM_kernel<<< blocks, threads >>>(A, B, C, *alpha, *beta, M, N, K, AT, BT);
    check_launch("myGEMM_kernel");
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

/* Routine for 10-class softmax */
void gpuSoftmax(double* A, unsigned int num_classes, unsigned int N) {
    unsigned int num_threads = 192;
    unsigned int thr_x = num_threads;
    dim3 threads(thr_x);

    unsigned int blk_x = (N + thr_x - 1) / thr_x;
    dim3 blocks(blk_x);

    gpuSoftmax_kernel<<< blocks, threads >>>(A, num_classes, N);
    check_launch("gpuSoftmax_kernel");
}

/* GPU kernel for in-place element-wise sigmoid */
__global__
void gpuSigmoid_kernel(double* A, unsigned int num_neurons, unsigned int N) {
    unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int col = blockIdx.y * blockDim.y + threadIdx.y;

    if(col < N && row < num_neurons) {
        int ij = row + (col * num_neurons);
        A[ij] = (double) 1.0 / (double)(1.0 + exp(-1.0 * A[ij]));
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

    gpuSigmoid_kernel<<< blocks, threads >>>(A, num_neurons, N);
    check_launch("gpuSigmoid_kernel");
}

/* GPU kernel for summing rows of matrix A. Places row sums in vector v*/
__global__
void gpuRowSum_kernel(double *A, double *v, int M, int N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M) {
        double rowSum = 0.0;
        for (int i = 0; i < N; i++) 
            rowSum += A[(M*i) + row];
        v[row] = rowSum;
    }
}

/* Routine for summing rows of matrix A. Places row sums in vector v */
void gpuRowSum(double *A, double *v, int M, int N) {
    unsigned int num_threads = 192;
    unsigned int thr_x = num_threads;
    dim3 threads(thr_x);

    unsigned int blk_x = (M + thr_x - 1) / thr_x;
    dim3 blocks(blk_x);

    gpuRowSum_kernel<<< blocks, threads >>>(A, v, M, N);
    check_launch("gpuRowSum_kernel");
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
void gpuElementwiseSum_kernel(double *A, double *B, double *C,
                              double alpha, double beta,
                              int M, int N) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (row < M && col < N) {
        int ind = row + (M*col);
        C[ind] = (alpha * A[ind]) + (beta * B[ind]);
    }
}

/* Routine for elementwise matrix sum */
void gpuElementwiseSum(double *A, double *B, double *C, 
                       double alpha, double beta,
                       int M, int N) {
    unsigned int num_threads = 192;
    unsigned int thr_x = 32;
    unsigned int thr_y = (num_threads + thr_x - 1) / thr_x;
    dim3 threads(thr_x, thr_y);

    unsigned int blk_x = (N + thr_x - 1) / thr_x;
    unsigned int blk_y = (M + thr_y - 1) / thr_y;
    dim3 blocks(blk_x, blk_y);

    gpuElementwiseSum_kernel<<< blocks, threads >>>(A, B, C, alpha, beta, M, N);
    check_launch("gpuElementwiseSum_kernel");
}

/* GPU kernel for derivative of sigmoid */
__global__
void gpudSigmoid_kernel(double *A, double *B, double *C, int M, int N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < M && col < N) {
        int ind = row + (M*col);
        C[ind] = (double) A[ind] * B[ind] * (1.0 - B[ind]);
    }
}

/** Routine for derivative of sigmoid */
void gpudSigmoid(double *A, double *B, double *C, int M, int N) {
    unsigned int num_threads = 192;
    unsigned int thr_x = 32;
    unsigned int thr_y = (num_threads + thr_x - 1) / thr_x;
    dim3 threads(thr_x, thr_y);

    unsigned int blk_x = (M + thr_x - 1) / thr_x;
    unsigned int blk_y = (N + thr_y - 1) / thr_y;
    dim3 blocks(blk_x, blk_y);

    gpudSigmoid_kernel<<< blocks, threads >>>(A, B, C, M, N);
    check_launch("gpudSigmoid");
}
/*
Routine to perform an in-place GEMM operation, i.e., C := alpha*A*B + beta*C
*/

/*
__global__
void myGEMMkernel(double* A, double* B, double* C, double alpha, double beta, int M,
           int N, int K,bool AT,bool BT) 
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (row < M && col < N) {
        int c_ind = row + (col * M);
        double dot_prod = 0.0;
        int a_ind;
        int b_ind;
        for(int i = 0; i < K; i++) {
            if (AT)
                a_ind = (row*K) + i;
            else
                a_ind = row + (i*M);
            if (BT)
                b_ind = col + (i*N);
            else
                b_ind = i + (col * K);
            dot_prod += A[a_ind] * B[b_ind];
        }
        C[c_ind] = (alpha * dot_prod) + (beta * C[c_ind]);
    }
    /*
    double inner_prod=0.0;
            if (AT){
			if(row< K&&col <N){
				for(int k = 0; k<K; k++){
                int indexA = (row*K)+k;
                int indexB = (col*K)+k;
				inner_prod+=A[indexA]*B[indexB];
            
				}
            }
			}
            else if(BT){
			if(row< M&&col <K){
				for(int k = 0; k<K; k++){
                int indexA = (k*M)+row;
                int indexB = (k*N)+col;
				inner_prod+=A[indexA]*B[indexB];
            }
			}
			}

            else{
			if(row< M&&col <N){
				for(int k = 0; k<K; k++){
					int indexA = (k*M)+row;
					int indexB = (col*K)+k;
					inner_prod+=A[indexA]*B[indexB];
            
            }
        }
    }
        C[col*M+row] =alpha*inner_prod+beta*C[col*M+row];
      */  

}

int myGEMM(double* A, double* B, double* C, double* alpha, double* beta, int M,
           int N, int K,bool AT,bool BT) 
{
    /* TODO: Write an efficient GEMM implementation on GPU */
    dim3 dimBlock(32,6);
    dim3 dimGrid((M+dimBlock.x-1)/dimBlock.x, (N+dimBlock.y-1)/dimBlock.y);

    myGEMMkernel<<<dimGrid, dimBlock>>>(A,B,C,*alpha,*beta,M,N,K,AT,BT);

    return 0;
}

__global__
void softmax_kernel(double* A,int M, int N)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < N) {
        double denominator = 0.0;

        for(int c = 0; c < M; c++){
            denominator += (double) std::exp(A[col*M + c]);
        }

        for(int c = 0; c < M; c++){
            int ij = c + (col * M);
            A[ij] = (double) std::exp(A[ij])/ (double) denominator;
        }
    }
    /*
    double denom = 0.0;
    if(col<N){
    for (int i =0;i<M;i++)
    {
        denom+=(double) std::exp(Z[col*M+i]);
    }
    for (int i =0;i<M;i++)
    {
        A[col*M+i]=std::exp(Z[col*M+i])/(double)denom;
    }
    }*/
}

void softmax_p(double* A,int M, int N)
{
    dim3 dimBlock(192);
    dim3 dimGrid((N+dimBlock.y-1)/dimBlock.x);
    softmax_kernel<<<dimGrid,dimBlock>>>(A,M,N);

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

void sigmoid_p(double* A,int M, int N)
{
    dim3 dimBlock(32,6);
    dim3 dimGrid((M+dimBlock.x-1)/dimBlock.x, (N+dimBlock.y-1)/dimBlock.y);
    sigmoid_kernel<<<dimGrid,dimBlock>>>(A,M,N);
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


void row_sum(double* W, double* Y, int M, int N)
{
    dim3 dimBlock(192);
    dim3 dimGrid((M+dimBlock.x-1)/dimBlock.x);
    row_sum_kernel<<<dimGrid,dimBlock>>>(W,Y,M,N);


}

__global__
void elem_mult_kernel(double* A, double* B, double* C,int M,int N)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if(row<M&&col<N)
    {
        C[(col*M+row)]=A[(col*M+row)]*B[(col*M+row)];
    }

}
void elem_mult(double* A, double* B, double* C,int M,int N)
{
    dim3 dimBlock(32,32);
    dim3 dimGrid((M+dimBlock.x-1)/dimBlock.x, (N+dimBlock.y-1)/dimBlock.y);
    elem_mult_kernel<<<dimGrid,dimBlock>>>(A,B,C,M,N);
}

__global__
void elem_add_kernel(double* A, double* B, double* C,double alpha, double beta,int M,int N)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if(row<M&&col<N)
    {
        C[(col*M+row)]=alpha*A[(col*M+row)]+beta*B[(col*M+row)];
    }

}

void elem_add(double* A, double* B, double* C,double alpha, double beta,int M,int N)
{
    dim3 dimBlock(32,32);
    dim3 dimGrid((M+dimBlock.x-1)/dimBlock.x, (N+dimBlock.y-1)/dimBlock.y);
    elem_add_kernel<<<dimGrid,dimBlock>>>(A,B,C,alpha,beta,M,N);
}

__global__
void elem_mod_kernel(double* A, double* B,double alpha, double beta,int M,int N)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if(row<M&&col<N)
    {
        B[(col*M+row)]=alpha-beta*A[(col*M+row)];
    }

}

void elem_mod(double* A, double* B,double alpha, double beta,int M,int N)
{
    dim3 dimBlock(32,32);
    dim3 dimGrid((M+dimBlock.x-1)/dimBlock.x, (N+dimBlock.y-1)/dimBlock.y);
    elem_mod_kernel<<<dimGrid,dimBlock>>>(A,B,alpha,beta,M,N);



}
/* GPU kernel for derivative of sigmoid */

__global__
void sigmoid_back_kernel(double *A, double *B, double *C, int M, int N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < M && col < N) {
        C[(M*col)+row]=A[(M*col)+row] * B[(M*col)+row] * (1.0 -B[(M*col)+row]);
    }

}

/** Routine for derivative of sigmoid */
void sigmoid_back(double *A, double *B, double *C, int M, int N) {
    dim3 dimBlock(32,6);
    dim3 dimGrid((M+dimBlock.x-1)/dimBlock.x, (N+dimBlock.y-1)/dimBlock.y);
    sigmoid_back_kernel<<< dimGrid, dimBlock >>>(A, B, C, M, N);
}
*/
