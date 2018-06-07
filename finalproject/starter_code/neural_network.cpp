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

        double* DB0;
        double* DB1;

        double* W0;
        double* W1;

        double* DW0;
        double* DW1;

        double* X;
        double* Yc;
        double* Y;

        double* B0;
        double* B1;

        double* DA0;
        double* DZ1;

        double* diff;

    dev_cache(int L0,int L1,int L2,int L3)
    {
		cudaMalloc((void**)&W0, sizeof(double) * L2 * L1);
		cudaMalloc((void**)&W1, sizeof(double) * L3 * L2);
		cudaMalloc((void**)&B0, sizeof(double) * L2 * L0);
		cudaMalloc((void**)&B1, sizeof(double) * L3 * L0); 

		cudaMalloc((void**)&DW0, sizeof(double) *L2 * L1);
		cudaMalloc((void**)&DW1, sizeof(double) *L3 * L2);
		cudaMalloc((void**)&DB0, sizeof(double) *L2 * L0);
		cudaMalloc((void**)&DB1, sizeof(double) *L3 * L0);

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

        cudaFree(DB0);
        cudaFree(DB1);

        cudaFree(W0);
        cudaFree(W1);

        cudaFree(DW0);
        cudaFree(DW1);

        cudaFree(X);
        cudaFree(Yc);
        cudaFree(Y);
        cudaFree(diff);

        cudaFree(B0);
        cudaFree(B1);
		
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


void gpuBackprop(device_cache &d, int N, double regularization, NeuralNetwork &nn, int num_processes) {
    int num_neurons = d.num_neurons;
    int num_classes = d.num_classes;
    int num_pixels = d.num_pixels;
    double reg = regularization/ (double) num_processes;
    double one = 1.0;
    double zero = 0.0;
    double Ninv_pos = 1.0/((double) N * (double) num_processes);
    double Ninv_neg = -1.0*Ninv_pos;

    //find difference between labels and predictions
    //............
    //std::cout << "NINVPOS = " << Ninv_pos << std::endl;
    //double* h_y_h = (double*) malloc(sizeof(double) * num_classes * N);
    //cudaMemcpy(h_y_h, d.yh, sizeof(double) * num_classes * N, cudaMemcpyDeviceToHost);
    //double* h_y = (double*) malloc(sizeof(double) * num_classes * N);
    //cudaMemcpy(h_y, d.y, sizeof(double) * num_classes * N, cudaMemcpyDeviceToHost);
    //for(int i = 0; i < 5; i++){
    //    std::cout << "i: " << i << ", true: " << h_y[i] << std::endl;
    //    std::cout << "i: " << i << ", pred: " << h_y_h[i] << std::endl;
    //}
    //............
    gpuElementwiseSum(d.yh, d.y, d.y_diff, Ninv_pos, Ninv_neg, num_classes, N);
       
    //double* h_y_diff = (double*) malloc(sizeof(double) * num_classes * N);
    //cudaMemcpy(h_y_diff, d.y_diff, sizeof(double) * num_classes * N, cudaMemcpyDeviceToHost);
    //for(int i = 0; i < 5; i++){
    //    std::cout << "i: " << i << ", diff: " << h_y_diff[i] << std::endl;
    //}

    //calculate dW2
    cudaMemcpy(d.dW2, d.W2, sizeof(double) * num_classes * num_neurons, cudaMemcpyDeviceToDevice);
    myGEMM(d.y_diff, d.A1, d.dW2, &one, &reg, num_classes, num_neurons, N, false, true);

    //calculate db2
    gpuRowSum(d.y_diff, d.db2, num_classes, N);


    //dA1
    myGEMM(d.W2, d.y_diff, d.dA1, &one, &zero, num_neurons, N, num_classes, true, false);
    
    //.............
    //double* h_A1 = (double*) malloc(sizeof(double) * num_classes * N);
    //cudaMemcpy(h_A1, d.A1, sizeof(double) * num_classes * N, cudaMemcpyDeviceToHost);
    //double* h_dA1 = (double*) malloc(sizeof(double) * num_classes * N);
    //cudaMemcpy(h_dA1, d.dA1, sizeof(double) * num_classes * N, cudaMemcpyDeviceToHost);
    //for(int i = 0; i < 5; i++){
    //    std::cout << "i: " << i << ", A1: " << h_A1[i]<< ", dA1: " << h_dA1[i] << std::endl;
    //}
    //.............
    
    //dZ1
    gpudSigmoid(d.dA1, d.A1, d.dZ1, num_neurons, N);
    
    //.............
    //double* h_dZ1 = (double*) malloc(sizeof(double) * num_classes * N);
    //cudaMemcpy(h_dZ1, d.dZ1, sizeof(double) * num_classes * N, cudaMemcpyDeviceToHost);
    //for(int i = 0; i < 5; i++){
    //    std::cout << "i: " << i << ", dZ1: " << h_dZ1[i] << std::endl;
    //}
    //............
    
    //calculate dW1
    cudaMemcpy(d.dW1, d.W1, sizeof(double) * num_neurons * num_pixels, cudaMemcpyDeviceToDevice);
    myGEMM(d.dZ1, d.X, d.dW1, &one, &reg, num_neurons, num_pixels, N, false, true);
    
    //calculate db1
    gpuRowSum(d.dZ1, d.db1, num_neurons, N);
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

    /* HINT: You can obtain a raw pointer to the memory used by Armadillo Matrices
       for storing elements in a column major way. Or you can allocate your own array
       memory space and store the elements in a row major way. Remember to update the
       Armadillo matrices in NeuralNetwork &nn of rank 0 before returning from the function. */
    int num_pixels = nn.H[0];
    int num_neurons = nn.H[1];
    int num_classes = nn.H[2];
    /* iter is a variable used to manage debugging. It increments in the inner loop
       and therefore goes from 0 to epochs*num_batches */
    int iter = 0;

    double *host_dW1 = (double *) malloc(sizeof(double) * num_neurons * num_pixels);
    double *host_dW2 = (double *) malloc(sizeof(double) * num_classes * num_neurons);
    double *host_db1 = (double *) malloc(sizeof(double) * num_neurons);
    double *host_db2 = (double *) malloc(sizeof(double) * num_classes);

    double *host_dW1_red = (double *) malloc(sizeof(double) * num_neurons * num_pixels);
    double *host_dW2_red = (double *) malloc(sizeof(double) * num_classes * num_neurons);
    double *host_db1_red = (double *) malloc(sizeof(double) * num_neurons);
    double *host_db2_red = (double *) malloc(sizeof(double) * num_classes);

    device_cache d(batch_size, num_pixels, num_classes, num_neurons);

    double *X_batch = (double *) malloc(sizeof(double) * num_pixels * batch_size);
    double *y_batch = (double *) malloc(sizeof(double) * num_classes * batch_size);

    // adjust learning rate and regularization for number of processes
    //double mod_reg = reg/(double)num_procs;

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
            int batch_start = batch * batch_size;
            int in_proc = std::min(batch_size, N - batch_start)/num_procs;
            
            MPI_SAFE_CALL(
                MPI_Scatter( 
                    X.colptr(batch_start),
                    num_pixels * in_proc,
                    MPI_DOUBLE,
                    X_batch,
                    num_pixels * in_proc,
                    MPI_DOUBLE,
                    0, 
                    MPI_COMM_WORLD
            ));
            MPI_SAFE_CALL(
                MPI_Scatter(
                    y.colptr(batch_start),
                    num_classes * in_proc,
                    MPI_DOUBLE,
                    y_batch,
                    num_classes * in_proc,
                    MPI_DOUBLE,
                    0,
                    MPI_COMM_WORLD
            ));

            //data transfer
            checkCudaErrors(cudaMemcpy(d.X, X_batch, sizeof(double) * num_pixels * in_proc, cudaMemcpyHostToDevice));
            checkCudaErrors(cudaMemcpy(d.y, y_batch, sizeof(double) * num_classes * in_proc, cudaMemcpyHostToDevice));
            checkCudaErrors(cudaMemcpy(d.W1, nn.W[0].memptr(), sizeof(double) * num_pixels * num_neurons, cudaMemcpyHostToDevice));
            checkCudaErrors(cudaMemcpy(d.b1, nn.b[0].memptr(), sizeof(double) * num_neurons, cudaMemcpyHostToDevice));
            checkCudaErrors(cudaMemcpy(d.W2, nn.W[1].memptr(), sizeof(double) * num_neurons * num_classes, cudaMemcpyHostToDevice));
            checkCudaErrors(cudaMemcpy(d.b2, nn.b[1].memptr(), sizeof(double) * num_classes, cudaMemcpyHostToDevice));

            gpuFeedforward(d, in_proc, nn);

            gpuBackprop(d, in_proc, reg, nn, num_procs);

            //copy gradients to host
            checkCudaErrors(cudaMemcpy(host_dW1, d.dW1, sizeof(double) * num_pixels * num_neurons, cudaMemcpyDeviceToHost));
            checkCudaErrors(cudaMemcpy(host_db1, d.db1, sizeof(double) * num_neurons, cudaMemcpyDeviceToHost));
            checkCudaErrors(cudaMemcpy(host_dW2, d.dW2, sizeof(double) * num_neurons * num_classes, cudaMemcpyDeviceToHost));
            checkCudaErrors(cudaMemcpy(host_db2, d.db2, sizeof(double) * num_classes, cudaMemcpyDeviceToHost));

            MPI_SAFE_CALL(
                MPI_Allreduce(
                    host_dW1,
                    host_dW1_red,
                    (num_pixels * num_neurons),
                    MPI_DOUBLE,
                    MPI_SUM,
                    MPI_COMM_WORLD
            ));
            MPI_SAFE_CALL(
                MPI_Allreduce(
                    host_db1,
                    host_db1_red,
                    (num_neurons),
                    MPI_DOUBLE,
                    MPI_SUM,
                    MPI_COMM_WORLD
            ));
            MPI_SAFE_CALL(
                MPI_Allreduce(
                    host_dW2,
                    host_dW2_red,
                    (num_neurons * num_classes),
                    MPI_DOUBLE,
                    MPI_SUM,
                    MPI_COMM_WORLD
            ));
            MPI_SAFE_CALL(
                MPI_Allreduce(
                    host_db2,
                    host_db2_red,
                    (num_classes),
                    MPI_DOUBLE,
                    MPI_SUM,
                    MPI_COMM_WORLD
            ));
            //gradient descent
            nn.W[0] = nn.W[0] - learning_rate * arma::mat(host_dW1_red, nn.W[0].n_rows, nn.W[0].n_cols);
            nn.b[0] = nn.b[0] - learning_rate * arma::colvec(host_db1_red, nn.b[0].n_rows, nn.b[0].n_cols);
            nn.W[1] = nn.W[1] - learning_rate * arma::mat(host_dW2_red, nn.W[1].n_rows, nn.W[1].n_cols);
            nn.b[1] = nn.b[1] - learning_rate * arma::colvec(host_db2_red, nn.b[1].n_rows, nn.b[1].n_cols);

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

    free(host_dW1);
    free(host_db1);
    free(host_dW2);
    free(host_db2);

    free(host_dW1_red);
    free(host_db1_red);
    free(host_dW2_red);
    free(host_db2_red);

    free(X_batch);
    free(y_batch);
    
    d.~device_cache();

    error_file.close();
}

/*
 * TODO
 * Train the neural network &nn of rank 0 in parallel. Your MPI implementation
 * should mainly be in this function.
 */
/*
void feedforward_gpu(dev_cache& c,int D0, int D1,int D2,int D3)
{
    

        std::vector<double> test_Y(D0*D3); 
        std::vector<double> test_A0(D0*D2); 
        std::vector<double> test_W0(D1*D2);
        std::vector<double> test_W1(D3*D2);
        std::vector<double> test_A1(D0*D2);
        std::vector<double> test_B1(D0*D3);

		//cudaMemcpy(&test_W0[0],cache.dW0,sizeof(double)* D1 * D2,cudaMemcpyDeviceToHost);
		//std::cout<< "feed W0: "<<test_W0[0] <<std::endl;  
        //std::vector<double> testZ0(D2*D0); 
		//dZ1=dB0
        double alpha=1.0;
        double beta=0.0;
		myGEMM(c.W0,c.X,c.A0,&alpha,&beta,D2,D0,D1);

		elem_add(c.A0,c.B0,c.A0,alpha,alpha,D2,D0);
		cudaMemcpy(&test_A0[0],c.A0,sizeof(double)* D2 * D0,cudaMemcpyDeviceToHost);
		std::cout<< "feed A0: "<<test_A0[0] <<std::endl;  
		
		//dB0=sigmoid(dB0)
		sigmoid_p(c.A0,D2,D0);
		//cudaMemcpy(&test_A0[0],c.dA0,sizeof(double)* D2 * D0,cudaMemcpyDeviceToHost);
		//std::cout<< "feed A0: "<<test_A0[0] <<std::endl;  

		//dA0=dB0
		//cudaMemcpy(dA0,dZ1,sizeof(double)*nn.H[1]*D0,cudaMemcpyDeviceToDevice);

		//dB1=dW1*dA0+dB1
		myGEMM(c.W1,c.A0,c.A1,&alpha,&beta,D3,D0,D2);
		elem_add(c.A1,c.B1,c.A1,alpha,alpha,D3,D0);
		cudaMemcpy(&test_A1[0],c.A1,sizeof(double)* D3 * D0,cudaMemcpyDeviceToHost);
		std::cout<< "A1: "<<test_A1[0]<<std::endl; 

		//dY=softmax(dB1)
		softmax_p(c.A1,D3,D0);
		cudaMemcpy(c.Yc,c.A1,sizeof(double)*D3*D0,cudaMemcpyDeviceToDevice);
        cudaMemcpy(&test_Y[0],c.Yc,sizeof(double)* D3 * D0,cudaMemcpyDeviceToHost);
        std::cout<< "feed Yc: "<<test_Y[0]<<std::endl; 
		//cudaMemcpy(&test_Y[0],c.dY,sizeof(double)* D3 * D0,cudaMemcpyDeviceToHost);
		//std::cout<< "Y: "<<test_Y[0]<<std::endl; 


}

void backprop_gpu(dev_cache& c,NeuralNetwork& nn,double reg,int D0,int batch_size)
{
   int D1 = nn.H[0];               // input feature dimension
   int D2 = nn.W[0].n_rows;        // hidden layer dimension
   int D3 = nn.W[1].n_rows;        // output layer dimension
   std::vector<double> test_Z1(D2*D0); 
   std::vector<double> test_W0(D2*D1); 
   std::vector<double> test_A0(D2*D0); 
   std::vector<double> test_W1(D2*D3); 
   std::vector<double> test_B1(D3); 
   std::vector<double> test_DB0(D2); 
   std::vector<double> test_Y(D3*D0); 

   std::cout<< "batch_size: "<<batch_size <<std::endl; 
   double alpha =1.0/((double) batch_size);
   double beta = -alpha;

   //cudaMemcpy(&test_Y[0],c.dY,sizeof(double)* D3 * D0,cudaMemcpyDeviceToHost);
   //std::cout<< "back Y: "<<test_Y[0]<<std::endl; 
   
   //cudaMemcpy(&test_Y[0],c.dYc,sizeof(double)* D3 * D0,cudaMemcpyDeviceToHost);
   //std::cout<< "back Yc: "<<test_Y[0]<<std::endl; 

   elem_add(c.Yc,c.Y,c.diff,alpha,beta,D3,D0);
   //cudaMemcpy(&test_Y[0],c.dA0,sizeof(double)* D3 * D0,cudaMemcpyDeviceToHost);
   //std::cout<< "back A0: "<<test_Y[0]<<std::endl; 


   alpha=1.0;
   beta = 0.0;
   
   cudaMemcpy(c.DW1,c.W1,sizeof(double)*D3*D2,cudaMemcpyDeviceToDevice);
   myGEMM(c.diff,c.A0,c.DW1,&alpha,&reg,D3,D2,D0,false,true);
   
   //cudaMemcpy(&test_diff[0],c.diff,sizeof(double)* D3 * D0,cudaMemcpyDeviceToHost);
   //std::cout<< "diff: "<<test_diff[100]<<std::endl; 
   //
   //dDW1=dW1
   //cudaMemcpy(&test_W1[0],c.dDW1,sizeof(double)* D3 * D2,cudaMemcpyDeviceToHost);
   //std::cout<< "DW1: "<<test_W1[0]<<std::endl; 

   //dDB1=rowsum(diff)
   row_sum(c.diff,c.DB1,D3,D0);
   cudaMemcpy(&test_B1[0],c.DB1,sizeof(double)* D3,cudaMemcpyDeviceToHost);
   std::cout<< "DB1: "<<test_B1[0]<<std::endl; 

   //dDA0=dW1.T*diff
   beta=0.0;
   myGEMM(c.W1,c.diff,c.DA0, &alpha, &beta,D3,D2,D0,true,false);
   cudaMemcpy(&test_A0[0],c.DA0,sizeof(double)*D2*D0,cudaMemcpyDeviceToHost);
   std::cout<< "DA0: "<<test_A0[0]<<std::endl; 

   //dA0=1-dA0
   alpha=1.0;
   beta=-1.0;
   
   sigmoid_back(c.DA0, c.A0, c.DZ1, D2, D0);

   //dW0=dZ1.T*dX.T+reg*dW0
   cudaMemcpy(c.DW0,c.W0,sizeof(double)*D1*D2,cudaMemcpyDeviceToDevice);

   myGEMM(c.DZ1,c.X,c.DW0,&alpha,&reg,D2,D0,D1,false,true);
   cudaMemcpy(&test_W0[0],c.DW0,sizeof(double)*D2*D1,cudaMemcpyDeviceToHost);
   std::cout<< "DW0: "<<test_W0[0]<<std::endl; 

   //dDB0=rowsum(dZ1)
   row_sum(c.DZ1,c.DB0,D2,D0);
   cudaMemcpy(&test_DB0[0],c.DB0,sizeof(double)*D2,cudaMemcpyDeviceToHost);
   std::cout<< "DB0: "<<test_DB0[0]<<std::endl; 

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
/*
    int iter = 0;
    int D1 = nn.H[0];        // input feature dimension
    int D2 = nn.H[1];        // hidden layer dimension
    int D3 = nn.H[2];        // output layer dimension
    int D0 = batch_size/num_procs;

    
	double *hdw0 = (double *) malloc(sizeof(double) * D2 * D1);
    double *hdw1 = (double *) malloc(sizeof(double) * D2*D3);
    double *hdb0 = (double *) malloc(sizeof(double) * D2);
    double *hdb1 = (double *) malloc(sizeof(double) * D3);
                
    double *hdw0_l = (double *) malloc(sizeof(double) * D2 * D1);
    double *hdw1_l = (double *) malloc(sizeof(double) * D2*D3);
    double *hdb0_l = (double *) malloc(sizeof(double) * D2);
    double *hdb1_l = (double *) malloc(sizeof(double) * D3);

    double* x_sub= (double *) malloc(sizeof(double) * D1 * D0); 
    double* y_sub=(double *) malloc(sizeof(double) * D3 * D0); 
    dev_cache cache(D0,D1,D2,D3);

    /*
    arma::mat hdw0(D2,D1);
    arma::mat hdw1(D3,D2);
    arma::mat hdb0(D2,1);
    arma::mat hdb1(D3,1);

    arma::mat hdw0_l(D2,D1);
    arma::mat hdw1_l(D3,D2);
    arma::mat hdb0_l(D2,1);
    arma::mat hdb1_l(D3,1);
	*/
/*
    for(int epoch = 0; epoch < epochs; ++epoch) {
        int num_batches = (N + batch_size - 1)/batch_size;
        for(int batch = 0; batch < num_batches-1; ++batch) {
            /*
             * Possible Implementation:
             * 1. subdivide input batch of images and `MPI_scatter()' to each MPI node
             * 2. compute each sub-batch of images' contribution to network coefficient updates
             * 3. reduce the coefficient updates and broadcast to all nodes with `MPI_Allreduce()'
             * 4. update local network coefficient at each node
             */
/*
            int batch_start = batch *batch_size;
            double reg2=reg/(double)num_procs;
            
            
            int num_elem=std::min(N-batch*batch_size,batch_size);
            
            D0=num_elem/num_procs;
            
            arma::mat b0_rep = arma::repmat(nn.b[0], 1, D0);
            arma::mat b1_rep = arma::repmat(nn.b[1], 1, D0);

            
            MPI_SAFE_CALL(MPI_Scatter(X.colptr(batch_start),D0*D1,MPI_DOUBLE
                                        ,x_sub,D0*D1,MPI_DOUBLE,0,MPI_COMM_WORLD));
                        
            MPI_SAFE_CALL(MPI_Scatter(y.colptr(batch_start),D0*D3,MPI_DOUBLE
                                        ,y_sub,D0*D3,MPI_DOUBLE,0,MPI_COMM_WORLD));


            //dX=X
            checkCudaErrors(cudaMemcpy(cache.X, x_sub, sizeof(double) * D1 * D0, cudaMemcpyHostToDevice));
            
            //dW0=W0
            checkCudaErrors(cudaMemcpy(cache.W0, nn.W[0].memptr(), sizeof(double) * D2 * D1, cudaMemcpyHostToDevice));

            //db0=b0
            checkCudaErrors(cudaMemcpy(cache.B0, b0_rep.memptr(), sizeof(double) * D2 * D0, cudaMemcpyHostToDevice));


            //dW1=W0
            checkCudaErrors(cudaMemcpy(cache.W1, nn.W[1].memptr(), sizeof(double) * D3 * D2, cudaMemcpyHostToDevice));

            //db1=b1
            checkCudaErrors(cudaMemcpy(cache.B1, b1_rep.memptr(), sizeof(double) * D3 * D0, cudaMemcpyHostToDevice));

            //dY= true Y's
            //checkCudaErrors(cudaMemcpy(cache.Y, y_sub, sizeof(double) * D3 * D0, cudaMemcpyHostToDevice));

            
            //dB0=dW0*dX+dB0
            feedforward_gpu(cache,D0,D1,D2,D3);
            
            backprop_gpu(cache,nn,reg2,D0,num_elem);


            //dYc=1/D0(dY-dYc)
            
            //dDW1=diff*dA0.T+reg*dDW1

            checkCudaErrors(cudaMemcpy(hdw0_l,cache.DW0,sizeof(double) * D2 * D1, cudaMemcpyDeviceToHost));
            checkCudaErrors(cudaMemcpy(hdw1_l,cache.DW1,sizeof(double) * D3 * D2, cudaMemcpyDeviceToHost));
            checkCudaErrors(cudaMemcpy(hdb0_l,cache.DB0,sizeof(double) * D2 , cudaMemcpyDeviceToHost));
            checkCudaErrors(cudaMemcpy(hdb1_l,cache.DB1,sizeof(double) * D3 , cudaMemcpyDeviceToHost));
           
            MPI_SAFE_CALL(MPI_Allreduce(hdw0_l,hdw0, D2 * D1, MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD));
            MPI_SAFE_CALL(MPI_Allreduce(hdw1_l,hdw1, D3 * D2, MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD));
            MPI_SAFE_CALL(MPI_Allreduce(hdb0_l,hdb0, D2, MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD));
            MPI_SAFE_CALL(MPI_Allreduce(hdb1_l,hdb1, D3, MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD));
            std::cout<< "W0 before:"<<nn.W[0][0]<<std::endl; 

            nn.W[0]=nn.W[0]-(learning_rate)*arma::mat(hdw0,nn.W[0].n_rows,nn.W[0].n_cols);
            nn.W[1]=nn.W[1]-(learning_rate)*arma::mat(hdw1,nn.W[1].n_rows,nn.W[1].n_cols);
            nn.b[0]=nn.b[0]-(learning_rate)*arma::mat(hdb0,nn.b[0].n_rows,nn.b[0].n_cols);
            nn.b[1]=nn.b[1]-(learning_rate)*arma::mat(hdb1,nn.b[1].n_rows,nn.b[1].n_cols);

            std::cout<< "W0 after:"<<nn.W[0][0]<<std::endl; 
            

            if(print_every <= 0) {
                print_flag = batch == 0;
            } else {
                print_flag = iter % print_every == 0;
            }

            /* Following debug routine assumes that you have already updated the arma
               matrices in the NeuralNetwork nn.  */
/*
            if(debug && rank == 0 && print_flag) {
                write_diff_gpu_cpu(nn, iter, error_file);
            }

            iter++;
        }
    }

    free(hdw0); 
    free(hdw1);
    free(hdb0);
    free(hdb1); 

    free(hdw0_l);
    free(hdw1_l);
    free(hdb0_l);
    free(hdb1_l);

    free(x_sub);
    free(y_sub);
    error_file.close();
}
