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
#include "vtk.h"

using namespace std;

// TODO: matrix assembly 2D 3D
    // NOTE: 2D assembly not symmetric
// TODO: store local k matrix in constant memory
// TODO: 3d elements' node distribution
// TODO: bc cases


// TODO: fix prolongation assembly - has something to do with bc initialization
// TODO: applyMatrixBC_GPU( valuevector, indexvector, mrs, bcindex(node), "which dimension is free", numrows)
// TODO: solve - ApplyTransposed() --> use restrictMatrix


int main()
{

    // Material properties
    double youngMod = 210e6;
    double poisson = 0.3;


    size_t numLevels = 3;

    // domain dimensions (x,y,z) on coarsest grid
    vector<size_t> N;
    vector<vector<size_t>> bc_index(numLevels);

    
    // base : {1,1}
    N = {1,1};
    bc_index[0] = { 0,1, 4,5 };
    bc_index[1] = { 0,1, 6,7, 12,13 };
    bc_index[2] = { 0,1, 10,11, 20,21, 30,31, 40,41 };

    // MBB : {2,1}
    // N = {2,1};
    // bc_index[0] = {0,1, 6,7};
    // bc_index[1] = {0,1, 10,11, 20,21};


    // MBB : {3,1}
    // N = {3,1};
    // bc_index[0] = {0,1, 8,9};
    // bc_index[1] = {0,1, 14,15, 28,29};


    size_t dim = N.size();
    
    // local element mesh size on coarsest grid
    double h_coarse = 1;
    
    // calculating the mesh size on the top level grid
    double h = h_coarse/pow(2,numLevels - 1);

    // smoother (jacobi damping parameter)
    double damp = 2.0/3.0;
    
    // boundary conditions (nodes)
    // TODO: give BC cases
        // MBB with fixed sides
        // MBB with ...
    // TODO: assembleBC( size_t case );

    



   
    // TDO
    double rho = 0.4;
    size_t p = 3;

    vector<size_t> num_rows;
    vector<size_t> max_row_size;
    vector<size_t> p_max_row_size;
    vector<size_t> r_max_row_size;

    //// device pointers

    // local stiffness
    double* d_A_local;

    // global stiffness matrix on each grid-level
    vector<double*> d_value;
    vector<size_t*> d_index;

    // prolongation matrices
    vector<double*> d_p_value;
    vector<size_t*> d_p_index;

    // restriction matrices
    vector<double*> d_r_value;
    vector<size_t*> d_r_index;

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
    Assembly.init(d_A_local, d_value, d_index, d_p_value, d_p_index, d_r_value, d_r_index, d_kai, num_rows, max_row_size, p_max_row_size, r_max_row_size, d_node_index);

    
//     /*
//     NOTE: after assembling you should have these :
//     global stiffness matrix ELLPACK
//         - vector<double*> d_value(numLevels)
//         - vector<size_t> d_index(numLevels)
//         - vector<size_t> max_row_size(numLevels)
//         - vector<double*> d_p_value(numLevels - 1)
//         - vector<size_t*> d_p_index(numLevels - 1)
//         - vector<size_t> p_max_row_size(numLevels -1 )
//     */
    
    // vector u, b
    vector<double> b(num_rows[numLevels - 1], 0);
    // b[5] = -10000;  // 1x1 base, 2 levels
    b[9] = -10000;  // 1x1 base, 3 levels

    // b[9] = -10000; // 2x1 base, 2 levels
    // b[13] = -10000; // 3x1 base, 2 levels
    double* d_u;
    double* d_b;


    CUDA_CALL( cudaMalloc((void**)&d_u, sizeof(double) * num_rows[numLevels - 1] ) );
    CUDA_CALL( cudaMalloc((void**)&d_b, sizeof(double) * num_rows[numLevels - 1] ) );

    CUDA_CALL( cudaMemset(d_u, 0, sizeof(double) * num_rows[numLevels - 1]) );

    CUDA_CALL( cudaMemcpy(d_b, &b[0], sizeof(double) * num_rows[numLevels - 1], cudaMemcpyHostToDevice) );

    /*
    ##################################################################
    #                           SOLVER                               #
    ##################################################################
    */

    // TODO: remove num_cols
        

    //   printELL_GPU<<<1,1>>> ( d_value[0], d_index[0], max_row_size[0], num_rows[0], num_rows[0]);
    // printELL_GPU<<<1,1>>> ( d_value[1], d_index[1], max_row_size[1], num_rows[1], num_rows[1]);
//     // printELL_GPU<<<1,1>>> ( d_value[2], d_index[2], max_row_size[2], num_rows[2], num_rows[2]);

//     // cout << "solver" << endl;

    Solver GMG(d_value, d_index, d_p_value, d_p_index, numLevels, num_rows, max_row_size, p_max_row_size, damp);

    GMG.init();
    GMG.set_verbose(false, false);
    GMG.set_num_prepostsmooth(3,3);
    GMG.set_convergence_params(10, 1e-99, 1e-10);
    GMG.set_bs_convergence_params(10, 1e-99, 1e-10);
    GMG.set_cycle('V');
    GMG.set_steps(15, 5); 
    
    GMG.solve(d_u, d_b, d_value);
    cudaDeviceSynchronize();

  
    // printVector_GPU<<<1,Assembly.getNumElements()>>>( d_kai, Assembly.getNumElements());
    // printVector_GPU<<<1,num_rows[numLevels - 1]>>>( d_u, num_rows[numLevels - 1]);
    
    
    // /*
    // ##################################################################
    // #                           TDO                                  #
    // ##################################################################
    // */
    
    // // // TDO algorithm, tdo.cu
    // // // produces updated d_kai
    // cout << "\n";
    // cout << "TDO" << endl;


    size_t local_num_rows = 4 * dim;
    
    

    // TODO: incorporate this in the beginning
    double etastar = 12.0;
    double betastar = 2.0 * pow(h,2);

    
    TDO tdo(d_u, d_kai, h, dim, betastar, etastar, Assembly.getNumElements(), local_num_rows, d_A_local, d_node_index, Assembly.getGridSize(), rho, numLevels, p);
    tdo.init();
    tdo.innerloop(d_u, d_kai);    // get updated d_kai


    // printVector_GPU<<<1,num_rows[numLevels - 1]>>>( d_u, num_rows[numLevels - 1]);
    // // TODO: vtk stuff
    // vector<double> kai(Assembly.getNumElements());

    // cout << kai.size() << endl;
    // CUDA_CALL( cudaMemcpy(&kai[0], d_kai, sizeof(double) * Assembly.getNumElements(), cudaMemcpyDeviceToHost) );

    // stringstream ss; 
    // ss << "test.vtk";
    // WriteVectorToVTK(kai, "kai", ss.str(), dim, Assembly.getNumNodesPerDim(), h, Assembly.getNumNodes() );

    // cudaDeviceSynchronize();
    // printELL_GPU<<<1,1>>> ( d_value[0], d_index[0], max_row_size[0], num_rows[0], num_rows[0]);
    // printELL_GPU<<<1,1>>> ( d_value[1], d_index[1], max_row_size[1], num_rows[1], num_rows[1]);

    cudaDeviceSynchronize();
    // DEBUG:
    // printVector_GPU<<<1,Assembly.getNumElements()>>>( d_kai, Assembly.getNumElements());

//     // // update stiffness matrix with new d_kai
//     // // TODO: get d_value, d_index and d_A_local from the class, 
//     // // in the end, it's only Update..(d_kai)

    // TODO: no need for R-matrix
    // Assembly.UpdateGlobalStiffness(d_kai, d_value, d_index, d_p_value, d_p_index, d_r_value, d_r_index, d_A_local);
    // cudaDeviceSynchronize();    



    // cudaDeviceSynchronize();
    // printVector_GPU<<<1,num_rows[numLevels - 1]>>>( d_u, num_rows[numLevels - 1]);
    // printELL_GPU<<<1,1>>> ( d_value[0], d_index[0], max_row_size[0], num_rows[0], num_rows[0]);
    // printELL_GPU<<<1,1>>> ( d_value[1], d_index[1], max_row_size[1], num_rows[1], num_rows[1]);
    // printELL_GPU<<<1,1>>> ( d_p_value[0], d_p_index[0], p_max_row_size[0], num_rows[1], num_rows[0]);
    // printELL_GPU<<<1,1>>> ( d_r_value[0], d_r_index[0], r_max_row_size[0], num_rows[0], num_rows[1]);


    ////////////////
    // ITERATION
    ////////////////


    // for ( int i = 1 ; i < 2 ; ++i )
    // {
    // // TODO: something's wrong with the solver for N = {3,1}
    // cout << "iteration " << i << endl;
    // cudaDeviceSynchronize();
    // GMG.reinit();
    // GMG.set_verbose(false, false);
    // GMG.set_steps(150, 5); 
    // GMG.solve(d_u, d_b, d_value);
    // cudaDeviceSynchronize();

    // // printVector_GPU<<<1,num_rows[numLevels - 1]>>>( d_u, num_rows[numLevels - 1]);

    // tdo.innerloop(d_u, d_kai);
    
    // cudaDeviceSynchronize();
    // printVector_GPU<<<1,Assembly.getNumElements()>>>( d_kai, Assembly.getNumElements());
    // cudaDeviceSynchronize();
    // cout << "\n";
    // }

    cudaDeviceSynchronize();
}

    // PTAP_GPU consider using 2d blocks? :
    // https://www.quantstart.com/articles/Matrix-Matrix-Multiplication-on-the-GPU-with-Nvidia-CUDA/

// print_GPU<<<1,1>>> ( d_res0 );