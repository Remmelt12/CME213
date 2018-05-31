#include "neural_network.h"

#include <armadillo>
#include "utils/common.h"
#include "gpu_func.h"
#include "mpi.h"
#include "iomanip"

#define MPI_SAFE_CALL( call ) do {                               \
    int err = call;                                              \
    if (err != MPI_SUCCESS) {                                    \
        fprintf(stderr, "MPI error %d in file '%s' at line %i",  \
               err, __FILE__, __LINE__);                         \
        exit(1);                                                 \
    } } while(0)

double norms(NeuralNetwork& nn) {
    double norm_sum = 0;

    for(int i = 0; i < nn.num_layers; ++i)  {
        norm_sum += arma::accu(arma::square(nn.W[i]));
    }

    return norm_sum;
}

void write_cpudata_tofile(NeuralNetwork& nn, int iter) {
    std::stringstream s;
    s << "Outputs/CPUmats/SequentialW0-" << iter << ".mat";
    nn.W[0].save(s.str(), arma::raw_ascii);
    std::stringstream t;
    t << "Outputs/CPUmats/SequentialW1-" << iter << ".mat";
    nn.W[1].save(t.str(), arma::raw_ascii);
    std::stringstream u;
    u << "Outputs/CPUmats/Sequentialb0-" << iter << ".mat";
    nn.b[0].save(u.str(), arma::raw_ascii);
    std::stringstream v;
    v << "Outputs/CPUmats/Sequentialb1-" << iter << ".mat";
    nn.b[1].save(v.str(), arma::raw_ascii);
}

void write_diff_gpu_cpu(NeuralNetwork& nn, int iter,
                        std::ofstream& error_file) {
    arma::mat A, B, C, D;

    std::stringstream s;
    s << "Outputs/CPUmats/SequentialW0-" << iter << ".mat";
    A.load(s.str(), arma::raw_ascii);
    double max_errW0 = arma::norm(nn.W[0]-A, "inf")/arma::norm(A, "inf");
    double L2_errW0  = arma::norm(nn.W[0]-A,2)/arma::norm(A,2);

    std::stringstream t;
    t << "Outputs/CPUmats/SequentialW1-" << iter << ".mat";
    B.load(t.str(), arma::raw_ascii);
    double max_errW1 = arma::norm(nn.W[1]-B, "inf")/arma::norm(B, "inf");
    double L2_errW1  = arma::norm(nn.W[1]-B,2)/arma::norm(B,2);

    std::stringstream u;
    u << "Outputs/CPUmats/Sequentialb0-" << iter << ".mat";
    C.load(u.str(), arma::raw_ascii);
    double max_errb0 = arma::norm(nn.b[0]-C, "inf")/arma::norm(C, "inf");
    double L2_errb0  = arma::norm(nn.b[0]-C,2)/arma::norm(C,2);

    std::stringstream v;
    v << "Outputs/CPUmats/Sequentialb1-" << iter << ".mat";
    D.load(v.str(), arma::raw_ascii);
    double max_errb1 = arma::norm(nn.b[1]-D, "inf")/arma::norm(D, "inf");
    double L2_errb1  = arma::norm(nn.b[1]-D,2)/arma::norm(D,2);

    int ow = 15;

    if(iter == 0) {
        error_file << std::left<< std::setw(ow) << "Iteration" << std::left<< std::setw(
                       ow) << "Max Err W0" << std::left << std::setw(ow) << "Max Err W1"
                   << std::left<< std::setw(ow) << "Max Err b0" << std::left<< std::setw(
                       ow) << "Max Err b1" << std::left << std::setw(ow) << "L2 Err W0" << std::left
                   << std::setw(ow) << "L2 Err W1" << std::left<< std::setw(
                       ow) << "L2 Err b0" << std::left<< std::setw(ow) << "L2 Err b1" << "\n";
    }

    error_file << std::left << std::setw(ow) << iter << std::left << std::setw(
                   ow) << max_errW0 << std::left << std::setw(ow) << max_errW1 <<
               std::left << std::setw(ow) << max_errb0 << std::left << std::setw(
                   ow) << max_errb1 << std::left<< std::setw(ow) << L2_errW0 << std::left <<
               std::setw(ow) << L2_errW1 << std::left << std::setw(ow) << L2_errb0 <<
               std::left<< std::setw(ow) << L2_errb1 << "\n";

}

/* CPU IMPLEMENTATIONS */
void feedforward(NeuralNetwork& nn, const arma::mat& X, struct cache& cache) {
    cache.z.resize(2);
    cache.a.resize(2);

    // std::cout << W[0].n_rows << "\n";tw
    assert(X.n_rows == nn.W[0].n_cols);
    cache.X = X;
    int N = X.n_cols;

    arma::mat z1 = nn.W[0] * X + arma::repmat(nn.b[0], 1, N);
    cache.z[0] = z1;

    arma::mat a1;
    sigmoid(z1, a1);
    cache.a[0] = a1;

    assert(a1.n_rows == nn.W[1].n_cols);
    arma::mat z2 = nn.W[1] * a1 + arma::repmat(nn.b[1], 1, N);
    cache.z[1] = z2;

    arma::mat a2;
    softmax(z2, a2);
    cache.a[1] = cache.yc = a2;
}

/*
 * Computes the gradients of the cost w.r.t each param.
 * MUST be called after feedforward since it uses the bpcache.
 * @params y : C x N one-hot column vectors
 * @params bpcache : Output of feedforward.
 * @params bpgrads: Returns the gradients for each param
 */
void backprop(NeuralNetwork& nn, const arma::mat& y, double reg,
              const struct cache& bpcache, struct grads& bpgrads) {
    bpgrads.dW.resize(2);
    bpgrads.db.resize(2);
    int N = y.n_cols;

    // std::cout << "backprop " << bpcache.yc << "\n";
    arma::mat diff = (1.0 / N) * (bpcache.yc - y);
    bpgrads.dW[1] = diff * bpcache.a[0].t() + reg * nn.W[1];
    bpgrads.db[1] = arma::sum(diff, 1);
    arma::mat da1 = nn.W[1].t() * diff;

    arma::mat dz1 = da1 % bpcache.a[0] % (1 - bpcache.a[0]);

    bpgrads.dW[0] = dz1 * bpcache.X.t() + reg * nn.W[0];
    bpgrads.db[0] = arma::sum(dz1, 1);
}

/*
 * Computes the Cross-Entropy loss function for the neural network.
 */
double loss(NeuralNetwork& nn, const arma::mat& yc, const arma::mat& y,
            double reg) {
    int N = yc.n_cols;
    double ce_sum = -arma::accu(arma::log(yc.elem(arma::find(y == 1))));

    double data_loss = ce_sum / N;
    double reg_loss = 0.5 * reg * norms(nn);
    double loss = data_loss + reg_loss;
    // std::cout << "Loss: " << loss << "\n";
    return loss;
}

/*
 * Returns a vector of labels for each row vector in the input
 */
void predict(NeuralNetwork& nn, const arma::mat& X, arma::rowvec& label) {
    struct cache fcache;
    feedforward(nn, X, fcache);
    label.set_size(X.n_cols);

    for(int i = 0; i < X.n_cols; ++i) {
        arma::uword row;
        fcache.yc.col(i).max(row);
        label(i) = row;
    }
}

/*
 * Computes the numerical gradient
 */
void numgrad(NeuralNetwork& nn, const arma::mat& X, const arma::mat& y,
             double reg, struct grads& numgrads) {
    double h = 0.00001;
    struct cache numcache;
    numgrads.dW.resize(nn.num_layers);
    numgrads.db.resize(nn.num_layers);

    for(int i = 0; i < nn.num_layers; ++i) {
        numgrads.dW[i].resize(nn.W[i].n_rows, nn.W[i].n_cols);

        for(int j = 0; j < nn.W[i].n_rows; ++j) {
            for(int k = 0; k < nn.W[i].n_cols; ++k) {
                double oldval = nn.W[i](j,k);
                nn.W[i](j, k) = oldval + h;
                feedforward(nn, X, numcache);
                double fxph = loss(nn, numcache.yc, y, reg);
                nn.W[i](j, k) = oldval - h;
                feedforward(nn, X, numcache);
                double fxnh = loss(nn, numcache.yc, y, reg);
                numgrads.dW[i](j, k) = (fxph - fxnh) / (2*h);
                nn.W[i](j, k) = oldval;
            }
        }
    }

    for(int i = 0; i < nn.num_layers; ++i) {
        numgrads.db[i].resize(nn.b[i].n_rows, nn.b[i].n_cols);

        for(int j = 0; j < nn.b[i].size(); ++j) {
            double oldval = nn.b[i](j);
            nn.b[i](j) = oldval + h;
            feedforward(nn, X, numcache);
            double fxph = loss(nn, numcache.yc, y, reg);
            nn.b[i](j) = oldval - h;
            feedforward(nn, X, numcache);
            double fxnh = loss(nn, numcache.yc, y, reg);
            numgrads.db[i](j) = (fxph - fxnh) / (2*h);
            nn.b[i](j) = oldval;
        }
    }
}

/*
 * Train the neural network &nn
 */
void train(NeuralNetwork& nn, const arma::mat& X, const arma::mat& y,
           double learning_rate, double reg,
           const int epochs, const int batch_size, bool grad_check, int print_every,
           int debug) {
    int N = X.n_cols;
    int iter = 0;
    int print_flag = 0;

    for(int epoch = 0 ; epoch < epochs; ++epoch) {
        int num_batches = (N + batch_size - 1)/batch_size;

        for(int batch = 0; batch < num_batches; ++batch) {
            int last_col = std::min((batch + 1)*batch_size-1, N-1);
            arma::mat X_batch = X.cols(batch * batch_size, last_col);
            arma::mat y_batch = y.cols(batch * batch_size, last_col);

            struct cache bpcache;
            feedforward(nn, X_batch, bpcache);

            struct grads bpgrads;
            backprop(nn, y_batch, reg, bpcache, bpgrads);

            if(print_every > 0 && iter % print_every == 0) {
                if(grad_check) {
                    struct grads numgrads;
                    numgrad(nn, X_batch, y_batch, reg, numgrads);
                    assert(gradcheck(numgrads, bpgrads));
                }

                std::cout << "Loss at iteration " << iter << " of epoch " << epoch << "/" <<
                          epochs << " = " << loss(nn, bpcache.yc, y_batch, reg) << "\n";
            }

            // Gradient descent step
            for(int i = 0; i < nn.W.size(); ++i) {
                nn.W[i] -= learning_rate * bpgrads.dW[i];
            }

            for(int i = 0; i < nn.b.size(); ++i) {
                nn.b[i] -= learning_rate * bpgrads.db[i];
            }

            /* Debug routine runs only when debug flag is set. If print_every is zero, it saves
               for the first batch of each epoch to avoid saving too many large files.
               Note that for the first time, you have to run debug and serial modes together.
               This will run the following function and write out files to CPUmats folder.
               In the later runs (with same parameters), you can use just the debug flag to
               output diff b/w CPU and GPU without running CPU version */
            if(print_every <= 0) {
                print_flag = batch == 0;
            } else {
                print_flag = iter % print_every == 0;
            }

            if(debug && print_flag) {
                write_cpudata_tofile(nn, iter);
            }

            iter++;
        }
    }
}

/*
 * TODO
 * Train the neural network &nn of rank 0 in parallel. Your MPI implementation
 * should mainly be in this function.
 */
void parallel_train(NeuralNetwork& nn, const arma::mat& X, const arma::mat& y,
                    double learning_rate, double reg,
                    const int epochs, const int batch_size, bool grad_check, int print_every,
                    int debug) {

    int rank, num_procs;
    MPI_SAFE_CALL(MPI_Comm_size(MPI_COMM_WORLD, &num_procs));
    MPI_SAFE_CALL(MPI_Comm_rank(MPI_COMM_WORLD, &rank));

    int N = (rank == 0)?X.n_cols:0;
    MPI_SAFE_CALL(MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD));
  
    std::ofstream error_file;
    error_file.open("Outputs/CpuGpuDiff.txt");
    int print_flag = 0;



	double* x_sub;
    x_sub=(double*)malloc(nn.H[0]*batch_size/num_procs*sizeof(double));

    double* y_sub;
    y_sub=(double*)malloc(nn.H[2]*batch_size/num_procs*sizeof(double));



    double* hdw0;
    hdw0 = (double*)malloc(nn.H[1]*nn.H[0]*sizeof(double));

    double* hdw1;
    hdw1 = (double*)malloc(nn.H[2]*nn.H[1]*sizeof(double));

    double* hdb0;
    hdb0 = (double*)malloc(nn.H[1]*1*sizeof(double));

    double* hdb1;
    hdb1 = (double*)malloc(nn.H[2]*1*sizeof(double));

    double* hdw0_l;
    hdw0 = (double*)malloc(nn.H[1]*nn.H[0]*sizeof(double));

    double* hdw1_l;
    hdw1 = (double*)malloc(nn.H[2]*nn.H[1]*sizeof(double));

    double* hdb0_l;
    hdb0 = (double*)malloc(nn.H[1]*1*sizeof(double));

    double* hdb1_l;
    hdb1 = (double*)malloc(nn.H[2]*1*sizeof(double));
    // std::cout << W[0].n_rows << "\n";tw

    /* HINT: You can obtain a raw pointer to the memory used by Armadillo Matrices
       for storing elements in a column major way. Or you can allocate your own array
       memory space and store the elements in a row major way. Remember to update the
       Armadillo matrices in NeuralNetwork &nn of rank 0 before returning from the function. */

    /* iter is a variable used to manage debugging. It increments in the inner loop
       and therefore goes from 0 to epochs*num_batches */
    int iter = 0;


    for(int epoch = 0; epoch < epochs; ++epoch) {
        int num_batches = (N + batch_size - 1)/batch_size;
        std::cout<< nn.W[0][1]<<std::endl; 

        for(int batch = 0; batch < num_batches; ++batch) {
			std::cout<< batch <<std::endl; 
            /*
             * Possible Implementation:
             * 1. subdivide input batch of images and `MPI_scatter()' to each MPI node
             * 2. compute each sub-batch of images' contribution to network coefficient updates
             * 3. reduce the coefficient updates and broadcast to all nodes with `MPI_Allreduce()'
             * 4. update local network coefficient at each node
             */
            int last_col = std::min((batch + 1)*batch_size-1, N-1);
            int pN = batch_size/num_procs;

            arma::mat X_batch = X.cols(batch * batch_size, last_col);
            arma::mat y_batch = y.cols(batch * batch_size, last_col);

            MPI_Scatter(X_batch.memptr(),batch_size/num_procs,MPI_DOUBLE,x_sub,batch_size/num_procs,MPI_DOUBLE,0,MPI_COMM_WORLD);
            MPI_Scatter(y_batch.memptr(),batch_size/num_procs,MPI_DOUBLE,y_sub,batch_size/num_procs,MPI_DOUBLE,0,MPI_COMM_WORLD);

            //std::cout<< "Got here."<<std::endl; 
            double* dW0;
            cudaMalloc((void**)&dW0, sizeof(double) * nn.H[1] * nn.H[0]);
            double* dX;
            cudaMalloc((void**)&dX, sizeof(double) * nn.H[0] * pN);
            double* dB0;
            cudaMalloc((void**)&dB0, sizeof(double) * nn.H[1] * pN);

            double alpha=1.0;
            cudaMemcpy(dX, x_sub, sizeof(double) * nn.H[0] * pN, cudaMemcpyHostToDevice);
            cudaMemcpy(dW0, nn.W[0].memptr(), sizeof(double) * nn.H[1] * nn.H[0], cudaMemcpyHostToDevice);
            cudaMemcpy(dB0, nn.b[0].memptr(), sizeof(double) * nn.H[1] * pN, cudaMemcpyHostToDevice);

            myGEMM(dW0,dX,dB0,&alpha,&alpha,nn.H[1],pN,nn.H[0]);
            //std::cout<< "Got here2."<<std::endl; 
            sigmoid_p(dB0,dB0,nn.H[1],pN);
            //std::cout<< "Got here3."<<std::endl; 


            double* dB1;
            cudaMalloc((void**)&dB1, sizeof(double) * nn.H[1] * pN);
            //std::cout<< "Got here4."<<std::endl; 

            double* dW1;
            cudaMalloc((void**)&dW1, sizeof(double) * nn.H[2] * nn.H[1]);

            cudaMemcpy(dW1, nn.W[1].memptr(), sizeof(double) * nn.H[1] * pN, cudaMemcpyHostToDevice);
            cudaMemcpy(dB1, nn.b[1].memptr(), sizeof(double) * nn.H[2] * pN, cudaMemcpyHostToDevice);

            //std::cout<< "Got here5."<<std::endl; 
            myGEMM(dW1,dB0,dB1,&alpha,&alpha,nn.H[2],pN,nn.H[1]);
            
            double* dY;
            cudaMalloc((void**)&dY, sizeof(double) * nn.H[2] * pN);

            //std::cout<< "Got here6."<<std::endl; 
            softmax_p(dB1,dY,nn.H[2],pN);

            double* dYc;
            cudaMalloc((void**)&dYc, sizeof(double) * nn.H[2] * pN);
            cudaMemcpy(dYc, y_sub, sizeof(double) * nn.H[2] * pN, cudaMemcpyHostToDevice);

            double* dOnes;
            double* dOnes2;
            double* dEye;
            double* dA0;
            double* dZ1;
            double* dDB0;
            double* dDB1;

            cudaMalloc((void**)&dDB1, sizeof(double) * nn.H[2] * 1);
            cudaMalloc((void**)&dDB0, sizeof(double) * nn.H[1] * 1);

            cudaMalloc((void**)&dZ1, sizeof(double) * nn.H[1] * pN);

            arma::mat Ones = arma::ones<arma::mat>(nn.H[2],nn.H[2]);
            cudaMalloc((void**)&dOnes, sizeof(double) * nn.H[2] * nn.H[2]);
            cudaMemcpy(dOnes, Ones.memptr(), sizeof(double) * nn.H[2] * nn.H[2], cudaMemcpyHostToDevice);
            arma::mat Ones2 = arma::ones<arma::mat>(nn.H[1],pN);
            cudaMalloc((void**)&dOnes2, sizeof(double) * nn.H[1] * pN);
            cudaMemcpy(dOnes2, Ones2.memptr(), sizeof(double) * nn.H[1] * pN, cudaMemcpyHostToDevice);
            arma::mat Eye = arma::eye<arma::mat>(nn.H[1],nn.H[1]);
            cudaMalloc((void**)&dEye, sizeof(double) * nn.H[1] * nn.H[1]);
            cudaMemcpy(dEye, Eye.memptr(), sizeof(double) * nn.H[1] * nn.H[1], cudaMemcpyHostToDevice);

            cudaMalloc((void**)&dA0, sizeof(double) * nn.H[1] * pN);
            cudaMemcpy(dA0, dB0, sizeof(double) * nn.H[1] * pN, cudaMemcpyDeviceToDevice);

            alpha =1.0/N;
            double beta =-1.0/N;
            myGEMM(dOnes,dYc,dY,&alpha,&beta,nn.H[2],pN,nn.H[2]);

            alpha =1.0;
            myGEMM(dY,dB0,dW1,&alpha,&reg,nn.H[2],pN,nn.H[1],false,true);

            row_sum(dY,dDB1,nn.H[2],nn.H[1]);

            beta=0.0;
            myGEMM(dW1,dY,dYc, &alpha, &beta,nn.H[2],N,nn.H[1],true,false);

            beta=-1.0;
            myGEMM(dEye,dOnes2,dA0,&alpha,&beta,nn.H[1],pN,nn.H[1]);

            elem_mult(dA0,dB0,dA0,nn.H[1],pN);
            elem_mult(dYc,dA0,dZ1,nn.H[1],pN);

            myGEMM(dZ1,dX,dW0,&alpha,&reg,nn.H[1],pN,nn.H[0],false,true);

            row_sum(dZ1,dB0,nn.H[1],pN);


            cudaMemcpy(hdw0_l,dW0,sizeof(double) * nn.H[1] * nn.H[0], cudaMemcpyDeviceToHost);
            cudaMemcpy(hdw1_l,dW1,sizeof(double) * nn.H[2] * nn.H[1], cudaMemcpyDeviceToHost);
            cudaMemcpy(hdb0_l,dB0,sizeof(double) * nn.H[1] * 1, cudaMemcpyDeviceToHost);
            cudaMemcpy(hdb1_l,dB1,sizeof(double) * nn.H[1] * 1, cudaMemcpyDeviceToHost);
            
/*
            MPI_Allreduce(hdw0_l,hdw0,nn.H[1] * nn.H[0],MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
            MPI_Allreduce(hdw1_l,hdw1,nn.H[2] * nn.H[1],MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
            MPI_Allreduce(hdb0_l,hdb0,nn.H[1],MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
            MPI_Allreduce(hdb1_l,hdb1,nn.H[2],MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
*/
            
            arma::mat temp_hdw0=arma::ones<arma::mat>(nn.H[1],nn.H[0]);
            arma::mat temp_hdw1=arma::ones<arma::mat>(nn.H[2],nn.H[1]);
            arma::mat temp_hdb0=arma::ones<arma::mat>(nn.H[1],1);
            arma::mat temp_hdb1=arma::ones<arma::mat>(nn.H[2],1);

            memcpy(temp_hdw0.memptr(),hdw0,sizeof(double)*nn.H[1]*nn.H[0]);
            memcpy(temp_hdw1.memptr(),hdw1,sizeof(double)*nn.H[2]*nn.H[1]);
            memcpy(temp_hdb0.memptr(),hdb0,sizeof(double)*nn.H[1]*1);
            memcpy(temp_hdb1.memptr(),hdb1,sizeof(double)*nn.H[2]*1);
            
            nn.W[0]-=learning_rate*temp_hdw0;
            nn.W[1]-=learning_rate*temp_hdw1;
            nn.b[0]-=learning_rate*temp_hdb0;
            nn.b[1]-=learning_rate*temp_hdb1;

            
            cudaFree(dW0);
            cudaFree(dW1);
            cudaFree(dB0);
            cudaFree(dB1);
            cudaFree(dY);
            cudaFree(dYc);
            cudaFree(dEye);
            cudaFree(dOnes);
            cudaFree(dOnes2);
            cudaFree(dZ1);
            cudaFree(dX);

            if(print_every <= 0) {
                print_flag = batch == 0;
            } else {
                print_flag = iter % print_every == 0;
            }

            /* Following debug routine assumes that you have already updated the arma
               matrices in the NeuralNetwork nn.  */
            if(debug && rank == 0 && print_flag) {
                write_diff_gpu_cpu(nn, iter, error_file);
            }

            iter++;
        }
    }

    error_file.close();
}
