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


int main()
{
    // Material properties
    double youngMod = 210e6;
    double poisson = 0.3;

    // domain dimensions
    size_t dim = 2;
    double h = 0.5;     // local element size

    // number of elements per dimension
    // size_t Nx = 1;
    // size_t Ny = 1;

    vector<size_t> N = {1,1};

    // multigrid precond
    size_t numLevels = 2;

    // smoother (jacobi damping parameter)
    double damp = 2.0/3.0;
    
    // boundary conditions
    // TODO: give BC cases
        // MBB with fixed sides
        // MBB with ...
    // TODO: assembleBC( size_t case );

    vector<vector<size_t>> bc_index(numLevels);
    bc_index[0] = {0, 1, 4, 5};
    bc_index[1] = {0, 1, 6, 7, 12, 13};

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

    // vector u, b
    vector<double> b(18, 0);
    b[5] = -10000;
    double* d_u;
    double* d_b;

    //// CUDA
    vector<size_t*> d_node_index;
    // d_node_index.resize(4);


    // TODO: get num_rows
    CUDA_CALL( cudaMalloc((void**)&d_u, sizeof(double) * 18 ) );
    CUDA_CALL( cudaMalloc((void**)&d_b, sizeof(double) * 18 ) );

    CUDA_CALL( cudaMemset(d_u, 0, sizeof(double) * 18) );

    CUDA_CALL( cudaMemcpy(d_b, &b[0], sizeof(double) * 18, cudaMemcpyHostToDevice) );
    
    Assembler Assembly(dim, h, N, youngMod, poisson, rho, p, numLevels);
    Assembly.setBC(bc_index);
    Assembly.init(d_A_local, d_value, d_index, d_p_value, d_p_index, d_kai, num_rows, max_row_size, p_max_row_size, d_node_index);

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
   
    /*
    ##################################################################
    #                           SOLVER                               #
    ##################################################################
    */



    // // TODO: remove num_cols
    // Solver GMG(d_value, d_index, d_p_value, d_p_index, numLevels, num_rows, max_row_size, p_max_row_size, damp);
    
    // GMG.init();
    // GMG.set_num_prepostsmooth(1,1);
    // GMG.set_convergence_params(1, 1e-99, 1e-10);
    // GMG.set_bs_convergence_params(1, 1e-99, 1e-10);
    // GMG.set_cycle('V');
    // GMG.set_steps(150, 50);
    // cudaDeviceSynchronize();
    // GMG.solve(d_u, d_b);
    // GMG.solve_(d_value, d_index, max_row_size, d_p_value, d_p_index, p_max_row_size, d_u, d_b, numLevels, num_rows);
    // cudaDeviceSynchronize();
    
    // GMG.deallocate();    

    /*
    ##################################################################
    #                           TDO                                  #
    ##################################################################
    */

    
    // TDO algorithm, tdo.cu
    // produces updated d_kai

    // converge?
    // double eta = 12.0;
    // double beta = 1.0;

    // TDO tdo(d_u, d_kai, h, dim, beta, eta, Assembly.getNumElements(), num_rows[0], d_A_local, d_node_index, N, del_t);
    // tdo.init();
    // tdo.innerloop();    // get updated d_kai


    // // update stiffness matrix with new d_kai
    // // TODO: get d_value, d_index and d_A_local from the class, 
    // // in the end, it's only Update..(d_kai)
    // Assembly.UpdateGlobalStiffness(d_kai, d_value, d_index, d_A_local);

    // printELL_GPU<<<1,1>>> ( d_value[1], d_index[1], max_row_size[1], num_rows[1], num_rows[1]);




}


// print_GPU<<<1,1>>> ( d_res0 );