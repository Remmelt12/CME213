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
        double* dA0;
        double* dZ1;
        double* dDB0;
        double* dDB1;
        double* dW0;
        double* dX;
        double* dB0;
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



    /*
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
    */
    for(int epoch = 0; epoch < epochs; ++epoch) {
        int num_batches = (N + batch_size - 1)/batch_size;

        for(int batch = 0; batch < num_batches; ++batch) {
        arma::mat hdw0(nn.H[1],nn.H[0]);
        arma::mat hdw1(nn.H[2],nn.H[1]);
        arma::mat hdb0(nn.H[1],1);
        arma::mat hdb1(nn.H[2],1);

        arma::mat hdw0_l(nn.H[1],nn.H[0]);
        arma::mat hdw1_l(nn.H[2],nn.H[1]);
        arma::mat hdb0_l(nn.H[1],1);
        arma::mat hdb1_l(nn.H[2],1);
            /*
             * Possible Implementation:
             * 1. subdivide input batch of images and `MPI_scatter()' to each MPI node
             * 2. compute each sub-batch of images' contribution to network coefficient updates
             * 3. reduce the coefficient updates and broadcast to all nodes with `MPI_Allreduce()'
             * 4. update local network coefficient at each node
             */
        int last_col = std::min((batch + 1)*batch_size-1, N-1);
            arma::mat X_batch = X.cols(batch * batch_size, last_col);
            arma::mat Y_batch = y.cols(batch * batch_size, last_col);

            int bsize = X_batch.n_cols;     // batch size
            int sbsize = bsize / num_procs; // HOW TO HANDLE bsize not divisable by num_procs
            int D0 = sbsize;                // subbatch size
            int pN = sbsize;
            int batch_start = batch *batch_size*nn.H[0];
            int D1 = X_batch.n_rows;        // input feature dimension
            int D2 = nn.W[0].n_rows;        // hidden layer dimension
            int D3 = nn.W[1].n_rows;        // output layer dimension

            double reg2=reg/num_procs_d;
            int num_elem=std::min(N-batch*batch_size,batch_size);

            std::vector<double> x_sub(sbsize*D1);

            std::vector<double> y_sub(sbsize*D3);

            
            MPI_Scatter(X.memptr()+batch_start,num_elem/num_procs*nn.H[0],MPI_DOUBLE
                                        ,&x_sub[0],num_elem/num_procs*nn.H[0],MPI_DOUBLE,0,MPI_COMM_WORLD);
                        
            MPI_Scatter(y.memptr()+batch_start,num_elem/num_procs,MPI_DOUBLE
                                        ,&y_sub[0],num_elem/num_procs,MPI_DOUBLE,0,MPI_COMM_WORLD);
            
            arma::mat b0_EXPANDED = arma::repmat(nn.b[0], 1, D0);
            arma::mat b1_EXPANDED = arma::repmat(nn.b[1], 1, D0);
            double* W0_ = nn.W[0].memptr();
            double* W1_ = nn.W[1].memptr();
            double* b0_ = b0_EXPANDED.memptr();
            double* b1_ = b1_EXPANDED.memptr();
            //std::cout<<"initial b: " << b1_EXPANDED[0]<<std::endl; 

            //std::cout<< "Got here."<<std::endl; 
            double* dOnes2= 0;
            double* dW0= 0;
            double* dW1= 0;
            double* dB0= 0;
            double* dB1= 0;
            double* dDW0= 0;
            double* dDW1= 0;
            double* dDB0= 0;
            double* dDB1= 0;
            double* dA0= 0;
            double* dDA0= 0;
            double* dZ1= 0;
            double* dDZ1= 0;
            double* dX= 0;
            double* dY= 0;
            double* dYc= 0;
            double* diff= 0;

            cudaMalloc((void**)&dW0, sizeof(double) * D2 * D1);
            cudaMalloc((void**)&dW1, sizeof(double) * D3 * D2);
            cudaMalloc((void**)&dB0, sizeof(double) * D2 * D0);
            cudaMalloc((void**)&dB1, sizeof(double) * D3 * D0); 

            cudaMalloc((void**)&dDW0, sizeof(double) *D2 * D1);
            cudaMalloc((void**)&dDW1, sizeof(double) *D3 * D2);
            cudaMalloc((void**)&dDB0, sizeof(double) *D2 * D0);
            cudaMalloc((void**)&dDB1, sizeof(double) *D3 * D0);

            cudaMalloc((void**)&dA0, sizeof(double) * D2 * D0); 
            cudaMalloc((void**)&dDA0, sizeof(double) * D2* D0);
            cudaMalloc((void**)&dZ1, sizeof(double) * D2 * D0);
            cudaMalloc((void**)&dDZ1, sizeof(double) * D2 * D0);
            cudaMalloc((void**)&dX, sizeof(double) * D1 * D0);
            cudaMalloc((void**)&dY, sizeof(double) * D3 * D0);
            cudaMalloc((void**)&dYc, sizeof(double) * D3 * D0);
            cudaMalloc((void**)&diff, sizeof(double) * D3 * D0);



            // stretch the b's int matrices.
            arma::mat temp_b0=arma::repmat(nn.b[0], 1, N);
            arma::mat temp_b1=arma::repmat(nn.b[1], 1, N);

            std::vector<double> test(2*2,2.0);
            //std::cout<<"initial: " << (test[0]) <<std::endl; 

            double* dtest = 0;
            std::vector<double> htest(2*2);
            
            cudaMalloc((void**)&dtest, sizeof(double) * 2 * 2);
            
            cudaMemcpy(dtest, &test[0], sizeof(double) * 2 * 2,
                    cudaMemcpyHostToDevice);

            //double a =1.0;
            //double b =0.0;
            //elem_mult(dtest,dtest,dtest,2,2);
            
            cudaMemcpy(&htest[0],dtest , sizeof(double) * 2 * 2,
                    cudaMemcpyDeviceToHost);

            //std::cout<< "sigmoid: "<< htest[0] <<std::endl; 

            cudaFree(dtest);



            // Get matrices of all ones and the Identity for later use.

            arma::mat Ones2 = arma::ones<arma::mat>(nn.H[1],pN);
            cudaMalloc((void**)&dOnes2, sizeof(double) * nn.H[1] * pN);
            cudaMemcpy(dOnes2, Ones2.memptr(), sizeof(double) * nn.H[1] * pN,
                    cudaMemcpyHostToDevice);
            
            
            //dX=X
            cudaMemcpy(dX, &x_sub[0], sizeof(double) * nn.H[0] * pN, cudaMemcpyHostToDevice);
            
            //dW0=W0
            cudaMemcpy(dW0, W0_, sizeof(double) * nn.H[1] * nn.H[0]
                    , cudaMemcpyHostToDevice);

            //db0=b0
            cudaMemcpy(dB0, b0_, sizeof(double) * nn.H[1] * pN
                    , cudaMemcpyHostToDevice);

            cudaMemcpy(b0_, dB0, sizeof(double) * nn.H[1] * pN
                    , cudaMemcpyDeviceToHost);

            //std::cout<<"device b: " << b1_EXPANDED[0]<<std::endl; 

            //dW1=W0
            cudaMemcpy(dW1, W1_, sizeof(double) * nn.H[2] * nn.H[1]
                    , cudaMemcpyHostToDevice);

            //db1=b1
            cudaMemcpy(dB1, b1_, sizeof(double) * nn.H[1] * pN
                    , cudaMemcpyHostToDevice);

            //dYc= true Y's
            cudaMemcpy(dYc, &y_sub[0], sizeof(double) * nn.H[2] * pN, cudaMemcpyHostToDevice);

            
            //dB0=dW0*dX+dB0
            double alpha = 1.0;
            double beta = 1.0;
            myGEMM(dW0,dX,dB0,&alpha,&beta,nn.H[1],pN,nn.H[0]);

            //dZ1=dB0
            cudaMemcpy(dZ1,dB0,sizeof(double)*nn.H[1]*pN,cudaMemcpyDeviceToDevice);

            
            //dB0=sigmoid(dB0)
            sigmoid_p(dZ1,dA0,nn.H[1],pN);

            //std::vector<double> test2(2*2,1.0) 

            //dA0=dB0
            cudaMemcpy(dA0,dZ1,sizeof(double)*nn.H[1]*pN,cudaMemcpyDeviceToDevice);

            //dB1=dW1*dA0+dB1
            myGEMM(dW1,dA0,dB1,&alpha,&alpha,nn.H[2],pN,nn.H[1]);

            //dY=softmax(dB1)
            softmax_p(dB1,dY,nn.H[2],pN);

            //dYc=1/pN(dY-dYc)
            alpha =1.0/(((double)pN)*num_procs_d);
            beta = -1.0/(((double)pN)*num_procs_d);

            elem_add(dY,dYc,diff,alpha,beta,nn.H[2],pN);

            alpha =1.0;
            
            //dDW1=diff.Y*dA0.T+reg*dDW1
            myGEMM(diff,dA0,dW1,&alpha,&reg2,nn.H[2],pN,nn.H[1],false,true);
            //
            //dDW1=dW1
            cudaMemcpy(dDW1,dW1,sizeof(double)*nn.H[2]*nn.H[1],cudaMemcpyDeviceToDevice);

            //dDB1=rowsum(diff)
            row_sum(diff,dDB1,nn.H[2],nn.H[1]);

            //dDA0=dW1*diff
            beta=0.0;
            myGEMM(dW1,diff,dDA0, &alpha, &beta,nn.H[2],N,nn.H[1],true,false);

            //dA0=1-dA0
            alpha=1.0;
            beta=-1.0;
            elem_mod(dA0,dA0,alpha,beta,nn.H[1],pN);

            //dA0=dA0%dB0
            elem_mult(dA0,dB0,dA0,nn.H[1],pN);

            //dZ1=dDA0%dA0
            elem_mult(dDA0,dA0,dZ1,nn.H[1],pN);

            //dW0=dZ1.T*dX.T+reg*dW0
            myGEMM(dZ1,dX,dW0,&alpha,&reg2,nn.H[1],pN,nn.H[0],false,true);

            //dDB0=rowsum(dZ1)
            row_sum(dZ1,dDB0,nn.H[1],pN);

            cudaMemcpy(hdw0_l.memptr(),dW0,sizeof(double) * nn.H[1] * nn.H[0], cudaMemcpyDeviceToHost);
            cudaMemcpy(hdw1_l.memptr(),dDW1,sizeof(double) * nn.H[2] * nn.H[1], cudaMemcpyDeviceToHost);
            cudaMemcpy(hdb0_l.memptr(),dDB0,sizeof(double) * nn.H[1] , cudaMemcpyDeviceToHost);
            cudaMemcpy(hdb1_l.memptr(),dDB1,sizeof(double) * nn.H[2] , cudaMemcpyDeviceToHost);
           
            MPI_Allreduce(hdw0_l.memptr(),hdw0.memptr(), nn.H[1] * nn.H[0], MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
            MPI_Allreduce(hdw1_l.memptr(),hdw1.memptr(), nn.H[2] * nn.H[1], MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
            MPI_Allreduce(hdb0_l.memptr(),hdb0.memptr(), nn.H[1], MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
            MPI_Allreduce(hdb1_l.memptr(),hdb1.memptr(), nn.H[2], MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);

            struct grads grads;
            grads.dW.resize(2);
            grads.db.resize(2);
            grads.dW[0] = hdw0;
            grads.dW[1] = hdw1;
            grads.db[0] = hdb0;
            grads.db[1] = hdb1;
            //std::cout<< hdw0[0]<<std::endl; 
            //std::cout<< hdw1[0]<<std::endl; 
            //std::cout<< hdb0[0]<<std::endl; 
            //std::cout<< hdb1[0]<<std::endl; 

            for(int i = 0; i < nn.W.size(); ++i) {
            //    std::cout<< "Before W" << i << ": " << nn.W[i][0] << "," << nn.W[i].n_rows<<std::endl; 
                nn.W[i] -= learning_rate * grads.dW[i];
            //    std::cout<< "After W" << i << ": " << nn.W[i][0] << "," << nn.W[i].n_rows<<std::endl; 
            }

            for(int i = 0; i < nn.b.size(); ++i) {
            //    std::cout<< "Before b" << i << ": " << nn.b[i].n_cols << "," << nn.b[i].n_rows<<std::endl; 
                nn.b[i] -= learning_rate * grads.db[i];
            //    std::cout<< "After b" << i << ": " << nn.b[i].n_cols << "," << nn.b[i].n_rows<<std::endl; 
            }

            /*
            nn.W[0]-=(learning_rate/num_procs_d)*hdw0;
            nn.W[1]-=(learning_rate/num_procs_d)*hdw1;
            nn.b[0]-=(learning_rate/num_procs_d)*hdb0;
            nn.b[1]-=(learning_rate/num_procs_d)*hdb1;
*/
            cudaFree(dOnes2);
            cudaFree(dW0);
            cudaFree(dW1);
            cudaFree(dB0);
            cudaFree(dB1);
            cudaFree(dDW0);
            cudaFree(dDW1);
            cudaFree(dDB0);
            cudaFree(dDB1);
            cudaFree(dA0);
            cudaFree(dDA0);
            cudaFree(dZ1);
            cudaFree(dDZ1);
            cudaFree(dX);
            cudaFree(dY);
            cudaFree(dYc);
            cudaFree(diff);

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
