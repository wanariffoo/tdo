/*
    
*/

#include <iostream>
#include <cuda.h>
#include <vector>
#include <cuda_runtime.h>
// #include "../include/mycudaheader.h"
// #include "precond.h"
#include "cudakernels.h"
#include "assemble.h"
#include "solver.h"
#include "tdo.h"

using namespace std;

// TODO: matrix assembly 2D 3D
// TODO: store local k matrix in constant memory
// TODO: fix prolongation assembly - has something to do with bc initialization
// TODO: 3d elements' node distribution
// TODO: bc cases
// TODO: work on 2D with 3 levels


int main()
{

    // Material properties
    double youngMod = 210e6;
    double poisson = 0.3;

    // domain dimensions (x,y,z)
    vector<size_t> N = {3,2,2};
    // vector<size_t> N = {1,1,1};
    size_t dim = N.size();
    double h = 0.5;     // local element mesh size


    // multigrid precond
    size_t numLevels = 3;

    // smoother (jacobi damping parameter)
    double damp = 2.0/3.0;
    
    // boundary conditions (nodes)
    // TODO: give BC cases
        // MBB with fixed sides
        // MBB with ...
    // TODO: assembleBC( size_t case );

    vector<vector<size_t>> bc_index(numLevels);
    bc_index[0] = {0,2};
    bc_index[1] = {0,3,6};
    bc_index[2] = {0,5,10,15,20};
   
    // TDO
    double rho = 0.4;
    size_t p = 3;
    double del_t = 1.0;

    vector<size_t> num_rows;
    vector<size_t> max_row_size;
    vector<size_t> p_max_row_size;

    //// device pointers

    // local stiffness
    double* d_A_local;
    // global stiffness matrix on each grid-level
    vector<double*> d_value;
    vector<size_t*> d_index;

    // prolongation matrices
    vector<double*> d_p_value;
    vector<size_t*> d_p_index;

    // design variable
    double* d_kai;           // NOTE: can alloc this immediately



    //// CUDA
    vector<size_t*> d_node_index;
    // d_node_index.resize(4);

    /*
    ##################################################################
    #                           ASSEMBLY                             #
    ##################################################################
    */
    
    Assembler Assembly(dim, h, N, youngMod, poisson, rho, p, numLevels);
    Assembly.setBC(bc_index);
    // Assembly.init(d_A_local, d_value, d_index, d_p_value, d_p_index, d_kai, num_rows, max_row_size, p_max_row_size, d_node_index);

    /*
    NOTE: after assembling you should have these :
    global stiffness matrix ELLPACK
        - vector<double*> d_value(numLevels)
        - vector<size_t> d_index(numLevels)
        - vector<size_t> max_row_size(numLevels)
        - vector<double*> d_p_value(numLevels - 1)
        - vector<size_t*> d_p_index(numLevels - 1)
        - vector<size_t> p_max_row_size(numLevels -1 )
    */
   
    // // vector u, b
    // vector<double> b(num_rows[numLevels - 1], 0);
    // b[10] = -10000;
    // double* d_u;
    // double* d_b;

    // // TODO: get num_rows
    // CUDA_CALL( cudaMalloc((void**)&d_u, sizeof(double) * num_rows[numLevels - 1] ) );
    // CUDA_CALL( cudaMalloc((void**)&d_b, sizeof(double) * num_rows[numLevels - 1] ) );

    // CUDA_CALL( cudaMemset(d_u, 0, sizeof(double) * num_rows[numLevels - 1]) );

    // CUDA_CALL( cudaMemcpy(d_b, &b[0], sizeof(double) * num_rows[numLevels - 1], cudaMemcpyHostToDevice) );

    /*
    ##################################################################
    #                           SOLVER                               #
    ##################################################################
    */

    // TODO: remove num_cols
        

    // printELL_GPU<<<1,1>>> ( d_value[0], d_index[0], max_row_size[0], num_rows[0], num_rows[0]);
    // printELL_GPU<<<1,1>>> ( d_value[1], d_index[1], max_row_size[1], num_rows[1], num_rows[1]);
    // printELL_GPU<<<1,1>>> ( d_value[2], d_index[2], max_row_size[2], num_rows[2], num_rows[2]);



    // Solver GMG(d_value, d_index, d_p_value, d_p_index, numLevels, num_rows, max_row_size, p_max_row_size, damp);

    // GMG.init();
    // GMG.set_num_prepostsmooth(3,3);
    // GMG.set_convergence_params(1, 1e-99, 1e-10);
    // GMG.set_bs_convergence_params(1, 1e-99, 1e-10);
    // GMG.set_cycle('V');
    // GMG.set_steps(20, 10); 
    // cudaDeviceSynchronize();
    // GMG.solve(d_u, d_b, d_value);
    // cudaDeviceSynchronize();

    // cudaDeviceSynchronize();

    // printVector_GPU<<<1,18>>>( d_u, 18);
    
    
    /*
    ##################################################################
    #                           TDO                                  #
    ##################################################################
    */

    
    // TDO algorithm, tdo.cu
    // produces updated d_kai

    // // TODO: incorporate this in the beginning
    // double eta = 12.0;
    // double beta = 1.0;

    // TDO tdo(d_u, d_kai, h, dim, beta, eta, Assembly.getNumElements(), num_rows[0], d_A_local, d_node_index, N, del_t);
    // tdo.init();
    // tdo.innerloop();    // get updated d_kai

    // // DEBUG:
    //     printVector_GPU<<<1,4>>>( d_kai, 4);

    // // update stiffness matrix with new d_kai
    // // TODO: get d_value, d_index and d_A_local from the class, 
    // // in the end, it's only Update..(d_kai)
    // Assembly.UpdateGlobalStiffness(d_kai, d_value, d_index, d_A_local);

    // printELL_GPU<<<1,1>>> ( d_value[numLevels - 1], d_index[numLevels - 1], max_row_size[numLevels - 1], num_rows[numLevels - 1], num_rows[numLevels - 1]);


    // GMG.reinit(); // TODO: update global matrix here, update the coarser ones here too 
    cudaDeviceSynchronize();

    
//     // TODO: remove d_value here
//     GMG.solve(d_u, d_b, d_value);


    
    
    
    // PTAP_GPU consider using 2d blocks? :
    // https://www.quantstart.com/articles/Matrix-Matrix-Multiplication-on-the-GPU-with-Nvidia-CUDA/



}


// print_GPU<<<1,1>>> ( d_res0 );