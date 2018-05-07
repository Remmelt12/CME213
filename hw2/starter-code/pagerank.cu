#include <iostream>
#include <string>
/* Each kernel handles the update of one pagerank score. In other
 * words, each kernel handles one row of the update:
 *
 *      pi(t+1) = (1/2) A pi(t) + (1 / (2N))
 *      
 * You may assume that num_nodes <= blockDim.x * 65535
 *
 */
__global__
void device_graph_propagate(const uint* graph_indices
                            , const uint* graph_edges
                            , const float* graph_nodes_in
                            , float* graph_nodes_out
                            , const float* inv_edges_per_node
                            , int num_nodes) {
    // TODO: fill in the kernel code here
    uint bid = blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;
    uint tid = bid*blockDim.x+threadIdx.x;
    float sum = 0.f;
    //for all of its edges
    if(tid<num_nodes){
        for(uint j = graph_indices[tid]; j < graph_indices[tid+1]; j++) {
            sum += graph_nodes_in[ graph_edges[j] ] * inv_edges_per_node[ graph_edges[j] ];
        }
        graph_nodes_out[tid] = 0.5f/(float)num_nodes + 0.5f*sum;
    }
}
#define checkCudaErrors(val) check( (val), #val, __FILE__, __LINE__)
template<typename T>
void check(T err, const char* const func, const char* const file, const int
        line) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error at: " << file << ":" << line <<std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func <<std::endl;
        exit(1);
    }
}
/* This function executes a specified number of iterations of the
 * pagerank algorithm. The variables are:
 *
 * h_graph_indices, h_graph_edges:
 *     These arrays describe the indices of the neighbors of node i.
 *     Specifically, node i is adjacent to all nodes in the range
 *     h_graph_edges[h_graph_indices[i] ... h_graph_indices[i+1]].
 *
 * h_node_values_input:
 *     An initial guess of pi(0).
 *
 * h_gpu_node_values_output:
 *     Output array for the pagerank vector.
 *
 * h_inv_edges_per_node:
 *     The i'th element in this array is the reciprocal of the
 *     out degree of the i'th node.
 *
 * nr_iterations:
 *     The number of iterations to run the pagerank algorithm for.
 *
 * num_nodes:
 *     The number of nodes in the whole graph (ie N).
 *
 * avg_edges:
 *     The average number of edges in the graph. You are guaranteed
 *     that the whole graph has num_nodes * avg_edges edges.
 *
 */
double device_graph_iterate(const uint* h_graph_indices
                            , const uint* h_graph_edges
                            , const float* h_node_values_input
                            , float* h_gpu_node_values_output
                            , const float* h_inv_edges_per_node
                            , int nr_iterations
                            , int num_nodes
                            , int avg_edges) {
    // TODO: allocate GPU memory
    uint* d_indices =0;
    uint* d_edges=0;
    float* d_buffer1=0;
    float* d_buffer2=0;
    float* d_inv_edges=0;

    checkCudaErrors(cudaMalloc(&d_indices,(num_nodes+1)*sizeof(uint)));
    
    checkCudaErrors(cudaMalloc(&d_edges,num_nodes*avg_edges*sizeof(uint)));
    
    checkCudaErrors(cudaMalloc(&d_buffer1,num_nodes*sizeof(float)));

    checkCudaErrors(cudaMalloc(&d_buffer2,num_nodes*sizeof(float)));
    
    checkCudaErrors(cudaMalloc(&d_inv_edges,num_nodes*sizeof(float)));


    // TODO: check for allocation failure

    // TODO: copy data to the GPU

    cudaMemcpy(d_buffer1,h_node_values_input,num_nodes*sizeof(float),cudaMemcpyHostToDevice);
    
    cudaMemcpy(d_indices,h_graph_indices,(num_nodes+1)*sizeof(uint),cudaMemcpyHostToDevice);
    
    cudaMemcpy(d_edges,h_graph_edges,avg_edges*num_nodes*sizeof(uint),cudaMemcpyHostToDevice);
    
    cudaMemcpy(d_inv_edges,h_inv_edges_per_node,num_nodes*sizeof(float),cudaMemcpyHostToDevice);

    
    start_timer(&timer);

    const int block_size = 192;

    // TODO: launch your kernels the appropriate number of iterations

    int num_blocks =(num_nodes + block_size - 1)/block_size; 
    int grid_x=std::min(num_blocks, 65535);
    int grid_y=std::min(std::max(1, num_blocks-grid_x), 65535);
    int grid_z=std::min(std::max(1, num_blocks-grid_x-grid_y), 65535);
    dim3 grid_size(grid_x,grid_y,grid_z);
    //std::cout<< grid_x<<" "<<grid_y<<" "<<grid_z<<std::endl; 

    for(int i=0;i<nr_iterations/2;i++)
    {
        device_graph_propagate<<<grid_size,block_size>>>( d_indices, d_edges, d_buffer1, d_buffer2
                            , d_inv_edges, num_nodes); 

        device_graph_propagate<<<grid_size,block_size>>>( d_indices, d_edges, d_buffer2, d_buffer1
                            , d_inv_edges, num_nodes); 
    }

    check_launch("gpu graph propagate");
    double gpu_elapsed_time = stop_timer(&timer);

    // TODO copy final data back to the host for correctness checking
    if(nr_iterations % 2) {
        device_graph_propagate<<<grid_size,block_size>>>( d_indices
                            , d_edges
                            , d_buffer1
                            , d_buffer2
                            , d_inv_edges
                            , num_nodes); 
        cudaMemcpy(h_gpu_node_values_output,d_buffer2,num_nodes*sizeof(float),cudaMemcpyDeviceToHost);
    } else {
        cudaMemcpy(h_gpu_node_values_output,d_buffer1,num_nodes*sizeof(float),cudaMemcpyDeviceToHost);
    }

    // TODO: free the memory you allocated!
    cudaFree(d_buffer1);
    cudaFree(d_buffer2);
    cudaFree(d_edges);
    cudaFree(d_inv_edges);
    cudaFree(d_indices);

    return gpu_elapsed_time;
}
