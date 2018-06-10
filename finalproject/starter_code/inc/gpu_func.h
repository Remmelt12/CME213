#ifndef GPU_FUNC_H_
#define GPU_FUNC_H_

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>

struct event_pair {
        cudaEvent_t start;
            cudaEvent_t end;
};

inline void check_launch(const char* kernel_name) {
        cudaThreadSynchronize();
            cudaError_t err = cudaGetLastError();

                if(err != cudaSuccess) {
                            std::cerr << "error in " << kernel_name << " kernel" << std::endl;
                                    std::cerr << "error was: " << cudaGetErrorString(err) <<std::endl;
                                            exit(1);
                                                }
}

inline void start_timer(event_pair* p) {
        cudaEventCreate(&p->start);
            cudaEventCreate(&p->end);
                cudaEventRecord(p->start, 0);
}


inline double stop_timer(event_pair* p) {
        cudaEventRecord(p->end, 0);
            cudaEventSynchronize(p->end);

                float elapsed_time;
                    cudaEventElapsedTime(&elapsed_time, p->start, p->end);
                        cudaEventDestroy(p->start);
                            cudaEventDestroy(p->end);
                                return elapsed_time;
}

int useless_gpu_add_one(int t);
int myGEMM(double* A, double* B, double* C,
           double* alpha, double* beta,
           int M, int N, int K,
           bool AT=false, bool BT=false);


void gpuSigmoid(double* A, unsigned int num_neurons, unsigned int N);
void gpuSoftmax(double* A, unsigned int num_classes, unsigned int N);
void gpuRowSum(double *A, double *v, int M, int N);
void gpuMatVecSum(double *A, double *v, int M, int N);
void gpuHadamard(double *A, double *B, double *C, int M, int N);
void gpuElementwiseSum(double *A, double *B, double *C, double alpha, double beta, int M, int N);
void gpuMatrixScalarProduct(double *A, double alpha, int M, int N);
void gpudSigmoid(double *A, double *B, double *C, int M, int N);

void sigmoid_p( double* A,int M, int N);

void softmax_p(double* A,int M, int N);

void row_sum(double* W, double* Y, int M, int N);

void elem_mult(double* A, double* B, double* C,int M,int N);

void elem_add(double* A, double* B, double* C,double alpha, double beta, int M,int N);

void elem_mod(double* A, double* B,double alpha, double beta, int M,int N);

void sigmoid_back(double *A, double *B, double *C, int M, int N);

#endif

