#include <math_constants.h>

#include "BC.h"

/**
 * Calculates the next finite difference step given a
 * grid point and step lengths.
 *
 * @param curr Pointer to the grid point that should be updated.
 * @param width Number of grid points in the x dimension.
 * @param xcfl Courant number for x dimension.
 * @param ycfl Courant number for y dimension.
 * @returns Grid value of next timestep.
 */
template<int order>
__device__
float Stencil(const float* curr, int width, float xcfl, float ycfl) {
    switch(order) {
        case 2:
            return curr[0] + xcfl * (curr[-1] + curr[1] - 2.f * curr[0]) +
                   ycfl * (curr[width] + curr[-width] - 2.f * curr[0]);

        case 4:
            return curr[0] + xcfl * (- curr[2] + 16.f * curr[1] - 30.f * curr[0] +
                                     16.f * curr[-1] - curr[-2]) + ycfl * (- curr[2 * width] +
                                             16.f * curr[width] - 30.f * curr[0] + 16.f * curr[-width] -
                                             curr[-2 * width]);

        case 8:
            return curr[0] + xcfl * (-9.f * curr[4] + 128.f * curr[3] -
                                     1008.f * curr[2] + 8064.f * curr[1] - 14350.f * curr[0] +
                                     8064.f * curr[-1] - 1008.f * curr[-2] + 128.f * curr[-3] -
                                     9.f * curr[-4]) + ycfl * (-9.f * curr[4 * width] +
                                             128.f * curr[3 * width] - 1008.f * curr[2 * width] +
                                             8064.f * curr[width] - 14350.f * curr[0] +
                                             8064.f * curr[-width] - 1008.f * curr[-2 * width] +
                                             128.f * curr[-3 * width] - 9.f * curr[-4 * width]);

        default:
            printf("ERROR: Order %d not supported", order);
            return CUDART_NAN_F;
    }
}

/**
 * Kernel to propagate finite difference grid from the current
 * time point to the next.
 *
 * This kernel should be very simple and only use global memory.
 *
 * @param next[out] Next grid state.
 * @param curr Current grid state.
 * @param gx Number of grid points in the x dimension.
 * @param nx Number of grid points in the x dimension to which the full
 *           stencil can be applied (ie the number of points that are at least
 *           order/2 grid points away from the boundary).
 * @param ny Number of grid points in the y dimension to which th full
 *           stencil can be applied.
 * @param xcfl Courant number for x dimension.
 * @param ycfl Courant number for y dimension.
 */
template<int order>
__global__
void gpuStencil(float* next,const float* curr, int gx, int nx, int ny,
                float xcfl, float ycfl) {
    // TODO
    uint i = blockIdx.x*blockDim.x+threadIdx.x;
    uint j = blockIdx.y*blockDim.y+threadIdx.y;

    uint bdr = (gx-nx)/2;
    if(i<nx && j<ny){
        uint index = i+bdr+gx*(j+bdr);
        next[index]=Stencil<order>(&curr[index],gx,xcfl,ycfl);
    }

}

/**
 * Propagates the finite difference 2D heat diffusion solver
 * using the gpuStencil kernel.
 *
 * Use this function to do necessary setup and propagate params.iters()
 * number of times.
 *
 * @param curr_grid The current state of the grid.
 * @param params Parameters for the finite difference computation.
 * @returns Time required for computation.
 */
double gpuComputation(Grid& curr_grid, const simParams& params) {

    boundary_conditions BC(params);

    Grid next_grid(curr_grid);

    // TODO: Declare variables/Compute parameters.

    const float xcfl = params.xcfl();
    const float ycfl = params.ycfl();


    const int nx = params.nx();
    const int ny = params.ny();

    const int order = params.order();
    dim3 threads(32, 6);
    dim3
        blocks((params.nx()+threads.x-1)/threads.x,(params.ny()+threads.y-1)/threads.y);

    const int gx = params.gx();
    std::cout<<blocks.x <<std::endl; 
    event_pair timer;
    start_timer(&timer);

    for(int i = 0; i < params.iters(); ++i) {

        // update the values on the boundary only
        BC.updateBC(next_grid.dGrid_, curr_grid.dGrid_);

        // TODO: Apply stencil.
        switch(order){
            case 2:
                gpuStencil<2><<<blocks,threads>>>(next_grid.dGrid_,
                                        curr_grid.dGrid_,gx,nx,ny,xcfl,ycfl); 
                break;
            case 4:
                gpuStencil<4><<<blocks,threads>>>(next_grid.dGrid_,
                                        curr_grid.dGrid_,gx,nx,ny,xcfl,ycfl); 
                break;
            case 8:
                gpuStencil<8><<<blocks,threads>>>(next_grid.dGrid_,
                                        curr_grid.dGrid_,gx,nx,ny,xcfl,ycfl); 
                break;
        }
        check_launch("gpuStencil");

        Grid::swap(curr_grid, next_grid);
    }

    return stop_timer(&timer);
}


/**
 * Kernel to propagate finite difference grid from the current
 * time point to the next.
 * This kernel should be optimized to compute finite difference updates
 * in blocks of size (blockDim.y * numYPerStep) * blockDim.x. Each thread
 * should calculate at most numYPerStep updates. It should still only use
 * global memory.
 *
 * @param next[out] Next grid state.
 * @param curr Current grid state.
 * @param gx Number of grid points in the x dimension.
 * @param nx Number of grid points in the x dimension to which the full
 *           stencil can be applied (ie the number of points that are at least
 *           order/2 grid points away from the boundary).
 * @param ny Number of grid points in the y dimension to which th full
 *           stencil can be applied.
 * @param xcfl Courant number for x dimension.
 * @param ycfl Courant number for y dimension.
 */
template<int order, int numYPerStep>
__global__
void gpuStencilLoop(float* next, const float* curr, int gx, int nx, int ny,
                    float xcfl, float ycfl) {
    // TODO
    uint i = blockIdx.x*blockDim.x+threadIdx.x;
    uint j = blockIdx.y*blockDim.y*numYPerStep+threadIdx.y;

    uint bdr = (gx-nx)/2;
    
    for (uint rowid=0;rowid<numYPerStep;rowid++){
        if(j<ny&&i<nx){ 
            uint index = i+bdr+gx*(j+bdr);
            next[index]=Stencil<order>(&curr[index],gx,xcfl,ycfl);
        }
        j+=blockDim.y;
    }    

}

/**
 * Propagates the finite difference 2D heat diffusion solver
 * using the gpuStencilLoop kernel.
 *
 * Use this function to do necessary setup and propagate params.iters()
 * number of times.
 *
 * @param curr_grid The current state of the grid.
 * @param params Parameters for the finite difference computation.
 * @returns Time required for computation.
 */
double gpuComputationLoop(Grid& curr_grid, const simParams& params) {
    boundary_conditions BC(params);

    Grid next_grid(curr_grid);
    // TODO
    // TODO: Declare variables/Compute parameters.
    const float xcfl = params.xcfl();
    const float ycfl = params.ycfl();
    const int num_y_steps=5;


    const int nx = params.nx();
    const int ny = params.ny();

    const int order = params.order();
    dim3 threads(32, 6);
    dim3
        blocks((params.nx()+threads.x-1)/threads.x,(params.ny()+num_y_steps*threads.y-1)/(threads.y*num_y_steps));

    const int gx = params.gx();
    event_pair timer;
    start_timer(&timer);

    for(int i = 0; i < params.iters(); ++i) {

        // update the values on the boundary only
        BC.updateBC(next_grid.dGrid_, curr_grid.dGrid_);

        // TODO: Apply stencil.
        switch(order){
            case 2:
                gpuStencilLoop<2,num_y_steps><<<blocks,threads>>>(next_grid.dGrid_,
                                        curr_grid.dGrid_,gx,nx,ny,xcfl,ycfl); 
                break;
            case 4:
                gpuStencilLoop<4,num_y_steps><<<blocks,threads>>>(next_grid.dGrid_,
                                        curr_grid.dGrid_,gx,nx,ny,xcfl,ycfl); 
                break;
            case 8:
                gpuStencilLoop<8,num_y_steps><<<blocks,threads>>>(next_grid.dGrid_,
                                        curr_grid.dGrid_,gx,nx,ny,xcfl,ycfl); 
                break;
        }
        check_launch("gpuStencilLoop");

        Grid::swap(curr_grid, next_grid);
    }

    return stop_timer(&timer);
}

/**
 * Kernel to propagate finite difference grid from the current
 * time point to the next.
 *
 * This kernel should be optimized to compute finite difference updates
 * in blocks of size side * side using shared memory.
 *
 * @param next[out] Next grid state.
 * @param curr Current grid state.
 * @param gx Number of grid points in the x dimension.
 * @param gy Number of grid points in the y dimension.
 * @param xcfl Courant number for x dimension.
 * @param ycfl Courant number for y dimension.
 */
template<int side, int order>
__global__
void gpuShared(float* next, const float* curr, int gx, int gy,
               float xcfl, float ycfl) {
    // TODO
    uint i = blockIdx.x*(blockDim.x-order)+threadIdx.x;
    uint j = blockIdx.y*(side-order)+threadIdx.y;

    uint bdr = order/2;

    __shared__ float submesh[side*side];

    for(int k =0;k<side/blockDim.y;k++){
        if( i<gx  && j<gy){
            uint index = i+gx*j;
            uint index_shared = threadIdx.x+side*(threadIdx.y+k*blockDim.y);
            submesh[index_shared]=curr[index];
        }
        j+=blockDim.y;
    }
    __syncthreads();
    
    j = blockIdx.y*(side-order)+threadIdx.y;
    
    uint left = ;
    uint right = side-bdr;
    uint top = bdr;
    uint bottom = side-bdr;


    for (uint k=0;k<side/blockDim.y;k++)
    {
        if((j<gx-bdr && i<gx-bdr)&&
            (bdr<=threadIdx.x && threadIdx.x<blockDim.x-bdr 
            && bdr <= threadIdx.y && threadIdx.y <blockDim.y-bdr))
        {

            uint index = i+gx*j;
            uint index_shared = threadIdx.x+side*(threadIdx.y+k*blockDim.y);
            next[index]=Stencil<order>(&submesh[index_shared],side,xcfl,ycfl);
            
        }
        j+=blockDim.y;
    }    

}

/**
 * Propagates the finite difference 2D heat diffusion solver
 * using the gpuShared kernel.
 *
 * Use this function to do necessary setup and propagate params.iters()
 * number of times.
 *
 * @param curr_grid The current state of the grid.
 * @param params Parameters for the finite difference computation.
 * @returns Time required for computation.
 */
template<int order>
double gpuComputationShared(Grid& curr_grid, const simParams& params) {

    boundary_conditions BC(params);

    Grid next_grid(curr_grid);

    // TODO: Declare variables/Compute parameters.
    const float xcfl = params.xcfl();
    const float ycfl = params.ycfl();
    const int side=64;


    const int gx = params.gx();
    const int gy = params.gy();

    dim3 threads(64, 8);
    dim3
        blocks((gx+threads.x-1)/threads.x,(gy+side-1)/side);


    event_pair timer;
    start_timer(&timer);


    for(int i = 0; i < params.iters(); ++i) {

        // update the values on the boundary only
        BC.updateBC(next_grid.dGrid_, curr_grid.dGrid_);
        switch(order){
            case 2:
                gpuShared<side,2><<<blocks,threads>>>(next_grid.dGrid_,
                                        curr_grid.dGrid_,gx,gy,xcfl,ycfl); 
                break;
            case 4:
                gpuShared<side,4><<<blocks,threads>>>(next_grid.dGrid_,
                                        curr_grid.dGrid_,gx,gy,xcfl,ycfl); 
                break;
            case 8:
                gpuShared<side,8><<<blocks,threads>>>(next_grid.dGrid_,
                                        curr_grid.dGrid_,gx,gy,xcfl,ycfl); 
                break;
        }

        // TODO: Apply stencil.

        check_launch("gpuShared");

        Grid::swap(curr_grid, next_grid);
    }

    return stop_timer(&timer);
}

