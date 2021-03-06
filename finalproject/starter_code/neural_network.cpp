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

struct dev_cache{
    public:
        double* A0;
        double* A1;
        double* Z1;
        double* Z0;

        double* Db0;
        double* Db1;

        double* W0;
        double* W1;

        double* DW0;
        double* DW1;

        double* X;
        double* Yc;
        double* Y;

        double* b0;
        double* b1;

        double* DA0;
        double* DZ1;

        double* diff;

    dev_cache(int L0,int L1,int L2,int L3)
    {
		cudaMalloc((void**)&W0, sizeof(double) * L2 * L1);
		cudaMalloc((void**)&W1, sizeof(double) * L3 * L2);
		cudaMalloc((void**)&b0, sizeof(double) * L2);
		cudaMalloc((void**)&b1, sizeof(double) * L3); 

		cudaMalloc((void**)&DW0, sizeof(double) *L2 * L1);
		cudaMalloc((void**)&DW1, sizeof(double) *L3 * L2);
		cudaMalloc((void**)&Db0, sizeof(double) *L2);
		cudaMalloc((void**)&Db1, sizeof(double) *L3);

		cudaMalloc((void**)&X, sizeof(double) * L1 * L0);

		cudaMalloc((void**)&A0, sizeof(double) * L2 * L0); 
		cudaMalloc((void**)&A1, sizeof(double) * L3 * L0); 
		cudaMalloc((void**)&Z1, sizeof(double) * L3 * L0);
		cudaMalloc((void**)&Z0, sizeof(double) * L2 * L0);

		cudaMalloc((void**)&DA0, sizeof(double) * L2 * L0); 
		cudaMalloc((void**)&DZ1, sizeof(double) * L2 * L0); 

		cudaMalloc((void**)&Y, sizeof(double) * L3 * L0);
		cudaMalloc((void**)&Yc, sizeof(double) * L3 * L0);
		cudaMalloc((void**)&diff, sizeof(double) * L3 * L0);
    }
	~dev_cache()
	{
        cudaFree(A0);
        cudaFree(A1);
        cudaFree(Z1);
        cudaFree(Z0);

        cudaFree(Db0);
        cudaFree(Db1);

        cudaFree(W0);
        cudaFree(W1);

        cudaFree(DW0);
        cudaFree(DW1);

        cudaFree(X);
        cudaFree(Yc);
        cudaFree(Y);
        cudaFree(diff);

        cudaFree(b0);
        cudaFree(b1);
		
        cudaFree(DA0);
        cudaFree(DZ1);
	
	}
};

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

void feedforward_gpu(dev_cache& d,NeuralNetwork& nn,double reg,int D0,int batch_size)
{
   int D1 = nn.H[0];               // input feature dimension
   int D2 = nn.W[0].n_rows;        // hidden layer dimension
   int D3 = nn.W[1].n_rows;        // output layer dimension
   double one = 1.0;
   double zero = 0.0;


   myGEMM(d.W0, d.X, d.A0, &one, &zero, D2, D0, D1);
   gpuMatVecSum(d.A0, d.b0, D2, D0);
   
   sigmoid_p(d.A0, D2, D0);
   //dB0=sigmoid(dB0)

   //dA0=dB0

   //dB1=dW1*dA0+dB1
   myGEMM(d.W1, d.A0, d.A1, &one, &zero, D3, D0, D2);

   //dY=softmax(dB1)
   gpuMatVecSum(d.A1, d.b1, D3, D0);
   softmax_p(d.A1, D3, D0);
   cudaMemcpy(d.Yc, d.A1, sizeof(double) * D3 * D0, cudaMemcpyDeviceToDevice);

   double alpha =1.0/((double) batch_size);
   double beta = -alpha;

   elem_add(d.Yc, d.Y, d.diff, alpha, beta, D3, D0);

   alpha=1.0;
   beta = 0.0;
   
   cudaMemcpy(d.DW1, d.W1, sizeof(double) * D3 * D2, cudaMemcpyDeviceToDevice);
   myGEMM(d.diff, d.A0, d.DW1, &one, &reg, D3, D2, D0, false, true);
   

   //dDB1=rowsum(diff)
   row_sum(d.diff, d.Db1, D3, D0);

   //dDA0=dW1.T*diff
   myGEMM(d.W1, d.diff, d.DA0, &one, &zero, D2, D0, D3, true, false);

   //dA0=1-dA0
   sigmoid_back(d.DA0, d.A0, d.DZ1, D2, D0);

   //dW0=dZ1.T*dX.T+reg*dW0
   cudaMemcpy(d.DW0, d.W0, sizeof(double) * D2 * D1, cudaMemcpyDeviceToDevice);
   myGEMM(d.DZ1, d.X, d.DW0, &one, &reg, D2, D1, D0, false, true);
   
   //calculate db1
   row_sum(d.DZ1, d.Db0, D2, D0);

}

void parallel_train(NeuralNetwork& nn, const arma::mat& X, const arma::mat& y,
                    double learning_rate, double reg,
                    const int epochs, const int batch_size, bool grad_check, int print_every,
                    int debug) {

    int rank, num_procs;
    MPI_SAFE_CALL(MPI_Comm_size(MPI_COMM_WORLD, &num_procs));
    MPI_SAFE_CALL(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    double num_procs_d=(double) num_procs;

    int N = (rank == 0)?X.n_cols:0;
    MPI_SAFE_CALL(MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD));
  
    std::ofstream error_file;
    error_file.open("Outputs/CpuGpuDiff.txt");
    int print_flag = 0;

    // std::cout << W[0].n_rows << "\n";tw

    /* HINT: You can obtain a raw pointer to the memory used by Armadillo Matrices
       for storing elements in a column major way. Or you can allocate your own array
       memory space and store the elements in a row major way. Remember to update the
       Armadillo matrices in NeuralNetwork &nn of rank 0 before returning from the function. */

    /* iter is a variable used to manage debugging. It increments in the inner loop
       and therefore goes from 0 to epochs*num_batches */

    int iter = 0;
    int D1 = nn.H[0];        // input feature dimension
    int D2 = nn.H[1];        // hidden layer dimension
    int D3 = nn.H[2];        // output layer dimension
    int D0 = batch_size/num_procs;
    
    std::vector<double>  x_sub(D1*D0); 
    std::vector<double>  y_sub(D3*D0); 
    dev_cache d(D0, D1, D2, D3);

    
    arma::mat hdw0(D2,D1);
    arma::mat hdw1(D3,D2);
    arma::mat hdb0(D2,1);
    arma::mat hdb1(D3,1);

    arma::mat hdw0_l(D2,D1);
    arma::mat hdw1_l(D3,D2);
    arma::mat hdb0_l(D2,1);
    arma::mat hdb1_l(D3,1);
    for(int epoch = 0; epoch < epochs; ++epoch) {
        int num_batches = (N + batch_size - 1)/batch_size;
        for(int batch = 0; batch < num_batches; ++batch) {
             /*
             * Possible Implementation:
             * 1. subdivide input batch of images and `MPI_scatter()' to each MPI node
             * 2. compute each sub-batch of images' contribution to network coefficient updates
             * 3. reduce the coefficient updates and broadcast to all nodes with `MPI_Allreduce()'
             * 4. update local network coefficient at each node
            */
            int batch_start = batch *batch_size;
            double reg2=reg/(double)num_procs;
            
            
            int num_elem=std::min(N-batch*batch_size,batch_size);
            
            D0=num_elem/num_procs;
            
            MPI_SAFE_CALL(MPI_Scatter(X.colptr(batch_start),D0*D1,MPI_DOUBLE
                                        ,&x_sub[0],D0*D1,MPI_DOUBLE,0,MPI_COMM_WORLD));
                        
            MPI_SAFE_CALL(MPI_Scatter(y.colptr(batch_start),D0*D3,MPI_DOUBLE
                                        ,&y_sub[0],D0*D3,MPI_DOUBLE,0,MPI_COMM_WORLD));

            //copy the data to the device
            checkCudaErrors(cudaMemcpy(d.X, &x_sub[0], sizeof(double) * D1 * D0, cudaMemcpyHostToDevice));
            checkCudaErrors(cudaMemcpy(d.Y, &y_sub[0], sizeof(double) * D3 * D0, cudaMemcpyHostToDevice));
            checkCudaErrors(cudaMemcpy(d.W0, nn.W[0].memptr(), sizeof(double) *D1 * D2, cudaMemcpyHostToDevice));
            checkCudaErrors(cudaMemcpy(d.b0, nn.b[0].memptr(), sizeof(double) * D2, cudaMemcpyHostToDevice));
            checkCudaErrors(cudaMemcpy(d.W1, nn.W[1].memptr(), sizeof(double) * D2 * D3, cudaMemcpyHostToDevice));
            checkCudaErrors(cudaMemcpy(d.b1, nn.b[1].memptr(), sizeof(double) * D3, cudaMemcpyHostToDevice));

            // feed forward and backprop in one.
            feedforward_gpu(d,nn,reg2,D0,num_elem);
            
            //copy the gradients back
            checkCudaErrors(cudaMemcpy(hdw0_l.memptr(), d.DW0, sizeof(double) * D1*D2, cudaMemcpyDeviceToHost));
            checkCudaErrors(cudaMemcpy(hdb0_l.memptr(), d.Db0, sizeof(double) * D2, cudaMemcpyDeviceToHost));
            checkCudaErrors(cudaMemcpy(hdw1_l.memptr(), d.DW1, sizeof(double) * D2*D3, cudaMemcpyDeviceToHost));
            checkCudaErrors(cudaMemcpy(hdb1_l.memptr(), d.Db1, sizeof(double) * D3, cudaMemcpyDeviceToHost));

            // Reduce
            MPI_SAFE_CALL(MPI_Allreduce(hdw0_l.memptr(),hdw0.memptr(), D2 * D1, MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD));
            MPI_SAFE_CALL(MPI_Allreduce(hdw1_l.memptr(),hdw1.memptr(), D3 * D2, MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD));
            MPI_SAFE_CALL(MPI_Allreduce(hdb0_l.memptr(),hdb0.memptr(), D2, MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD));
            MPI_SAFE_CALL(MPI_Allreduce(hdb1_l.memptr(),hdb1.memptr(), D3, MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD));

            //Gradient decent step
            nn.W[0]=nn.W[0]-(learning_rate)*hdw0;
            nn.W[1]=nn.W[1]-(learning_rate)*hdw1;
            nn.b[0]=nn.b[0]-(learning_rate)*hdb0;
            nn.b[1]=nn.b[1]-(learning_rate)*hdb1;

            

            if(print_every <= 0) {
                print_flag = batch == 0;
            } else {
                print_flag = iter % print_every == 0;
            }
/*
             Following debug routine assumes that you have already updated the arma
               matrices in the NeuralNetwork nn.  
*/
            if(debug && rank == 0 && print_flag) {
                write_diff_gpu_cpu(nn, iter, error_file);
            }

            iter++;
        }
    }

    error_file.close();
}

