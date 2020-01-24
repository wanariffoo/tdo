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

// DONE: bc cases
// DONE: matrix assembly 2D 3D
// DONE: fix prolongation assembly - has something to do with bc initialization
    // DONE: 2D assembly not symmetric

// TODO: 3d elements' node distribution
// TODO: store local k matrix in constant memory
// TODO: ApplyTranspose(prol) --> Apply(rest)
// TODO: applyMatrixBC_GPU( valuevector, indexvector, mrs, bcindex(node), "which dimension is free", numrows)
// TODO: laplacian_GPU() in 3D
    

int main()
{
  
    // create vtk files
    bool writeToVTK = true;

    // Material properties
    double youngMod = 210e6;
    double poisson = 0.3;


    size_t numLevels = 2;

    // domain dimensions (x,y,z) on coarsest grid
    vector<size_t> N;
    vector<vector<size_t>> bc_index(numLevels);

    //// 2D
    // base : {1,1}
    // N = {1,1};
    // bc_index[0] = { 0,1, 4,5 };
    // bc_index[1] = { 0,1, 6,7, 12,13 };
    // bc_index[2] = { 0,1, 10,11, 20,21, 30,31, 40,41 };
    // bc_index[3] = { 0,1, 18,19, 36,37, 54,55, 72,73, 90,91, 108,109, 126,127, 144,145};

    // MBB : {2,1}
    // N = {2,1};
    // bc_index[0] = {0,1, 6,7};
    // bc_index[1] = {0,1, 10,11, 20,21};


    // MBB : {3,1}
    N = {3,1};
    bc_index[0] = {0,7,8};
    bc_index[1] = {0,13,14,28};


    size_t dim = N.size();
    
    // local element mesh size on coarsest grid
    double h_coarse = 1;
    
    // calculating the mesh size on the top level grid
    double h = h_coarse/pow(2,numLevels - 1);

    // smoother (jacobi damping parameter)
    double damp = 2.0/3.0;

    size_t local_num_rows = 4 * dim;

    // TDO
    double rho = 0.4;
    size_t p = 3;
    double etastar = 12.0;
    double betastar = 2.0 * pow(h,2);

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
    double* d_chi;           // NOTE: can alloc this immediately



    //// CUDA
    vector<size_t*> d_node_index;
    // d_node_index.resize(4);

    // cout << "### GPU-accelerated Thermodynamic Topology Optimization ###" << endl;
    // cout << "Levels: " << numLevels << endl;

    /*
    ##################################################################
    #                           ASSEMBLY                             #
    ##################################################################
    */
    
    

    Assembler Assembly(dim, h, N, youngMod, poisson, rho, p, numLevels);
    Assembly.setBC(bc_index);
    Assembly.init(d_A_local, d_value, d_index, d_p_value, d_p_index, d_r_value, d_r_index, d_chi, num_rows, max_row_size, p_max_row_size, r_max_row_size, d_node_index);

    
    // vector u, b
    vector<double> b(num_rows[numLevels - 1], 0);
    // b[5] = -10000;  // 1x1 base, 2 levels
    // b[9] = -10000;  // 1x1 base, 3 levels
    // b[17] = -10000;  // 1x1 base, 4 levels

    // MBB
    b[29] = -10000; // 3x1 base, 2 levels


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
   

    // printELL_GPU<<<1,1>>> ( d_value[0], d_index[0], max_row_size[0], num_rows[0], num_rows[0]);
    // printELL_GPU<<<1,1>>> ( d_value[1], d_index[1], max_row_size[1], num_rows[1], num_rows[1]);
    // printELL_GPU<<<1,1>>> ( d_value[2], d_index[2], max_row_size[2], num_rows[2], num_rows[2]);
    // printELL_GPU<<<1,1>>> ( d_value[3], d_index[3], max_row_size[3], num_rows[3], num_rows[3]);


    Solver GMG(d_value, d_index, d_p_value, d_p_index, numLevels, num_rows, max_row_size, p_max_row_size, damp);
    GMG.set_convergence_params(500, 1e-99, 1e-10);
    // GMG.set_bs_convergence_params(10, 1e-99, 1e-10); // TODO: KIV
    

    GMG.init();
    GMG.set_verbose(0, 0);
    GMG.set_num_prepostsmooth(3,3);
    GMG.set_cycle('V');
    // GMG.set_steps(15, 5); 
    
    GMG.solve(d_u, d_b, d_value);
    cudaDeviceSynchronize();

  
    // printVector_GPU<<<1,Assembly.getNumElements()>>>( d_chi, Assembly.getNumElements());
    // printVector_GPU<<<1,num_rows[numLevels - 1]>>>( d_u, num_rows[numLevels - 1]);
    
    
    /*
    ##################################################################
    #                           TDO                                  #
    ##################################################################
    */

    
    TDO tdo(d_u, d_chi, h, dim, betastar, etastar, Assembly.getNumElements(), local_num_rows, d_A_local, d_node_index, Assembly.getGridSize(), rho, numLevels, p);
    tdo.init();
    tdo.innerloop(d_u, d_chi);    // get updated d_chi


    // printVector_GPU<<<1,num_rows[numLevels - 1]>>>( d_u, num_rows[numLevels - 1]);


    // TODO: write a function for this to make it neater
    // vtk stuff
    vector<double> chi(Assembly.getNumElements(), rho);
    string fileformat(".vtk");
    int file_index = 0;
    stringstream ss; 
    ss << "vtk/tdo";
    ss << file_index;
    ss << fileformat;

    if ( writeToVTK )
    {
        WriteVectorToVTK(chi, "chi", ss.str(), dim, Assembly.getGridSize(), h, Assembly.getNumElements() );
        
        CUDA_CALL( cudaMemcpy(&chi[0], d_chi, sizeof(double) * Assembly.getNumElements(), cudaMemcpyDeviceToHost) );

        file_index++;
        ss.str( string() );
        ss.clear();
        ss << "vtk/tdo";
        ss << file_index;
        ss << fileformat;
        
        WriteVectorToVTK(chi, "chi", ss.str(), dim, Assembly.getGridSize(), h, Assembly.getNumElements() );
    }




//     // cudaDeviceSynchronize();
    // printELL_GPU<<<1,1>>> ( d_value[0], d_index[0], max_row_size[0], num_rows[0], num_rows[0]);
    // printELL_GPU<<<1,1>>> ( d_value[1], d_index[1], max_row_size[1], num_rows[1], num_rows[1]);

//     cudaDeviceSynchronize();
    // DEBUG:
    // printVector_GPU<<<1,Assembly.getNumElements()>>>( d_chi, Assembly.getNumElements());


    // TODO: no need for R-matrix
    // Assembly.UpdateGlobalStiffness(d_chi, d_value, d_index, d_p_value, d_p_index, d_r_value, d_r_index, d_A_local);
    // cudaDeviceSynchronize();    


    // DEBUG:
    // // cudaDeviceSynchronize();
    // printVector_GPU<<<1,num_rows[numLevels - 1]>>>( d_u, num_rows[numLevels - 1]);
    // printVector_GPU<<<1,Assembly.getNumElements()>>>( d_chi, Assembly.getNumElements());

    // A matrix
    // printELL_GPU<<<1,1>>> ( d_value[0], d_index[0], max_row_size[0], num_rows[0], num_rows[0]);
    // printELL_GPU<<<1,1>>> ( d_value[1], d_index[1], max_row_size[1], num_rows[1], num_rows[1]);
    // printELL_GPU<<<1,1>>> ( d_value[2], d_index[2], max_row_size[2], num_rows[2], num_rows[2]);

    // prolongation matrix
    // printELL_GPU<<<1,1>>> ( d_p_value[0], d_p_index[0], p_max_row_size[0], num_rows[1], num_rows[0]);
    // printELL_GPU<<<1,1>>> ( d_p_value[1], d_p_index[1], p_max_row_size[1], num_rows[2], num_rows[1]);

    // restriction matrix
    // printELL_GPU<<<1,1>>> ( d_r_value[0], d_r_index[0], r_max_row_size[0], num_rows[0], num_rows[1]);
    // printELL_GPU<<<1,1>>> ( d_r_value[1], d_r_index[1], r_max_row_size[1], num_rows[1], num_rows[2]);


    ////////////////
    // ITERATION
    ////////////////


    for ( int i = 1 ; i < 50 ; ++i )
    {
    // TODO: something's wrong with the solver for N = {3,1}
    // cout << "iteration " << i << endl;
    cudaDeviceSynchronize();
    GMG.reinit();
    GMG.set_verbose(0, 0);
    // GMG.set_steps(200, 5); 
    GMG.solve(d_u, d_b, d_value);
    cudaDeviceSynchronize();

    // printVector_GPU<<<1,num_rows[numLevels - 1]>>>( d_u, num_rows[numLevels - 1]);


    tdo.innerloop(d_u, d_chi);
    
    // cudaDeviceSynchronize();
    // printVector_GPU<<<1,Assembly.getNumElements()>>>( d_chi, Assembly.getNumElements());
    // cout << "\n";

    if ( writeToVTK )
    {
        CUDA_CALL( cudaMemcpy(&chi[0], d_chi, sizeof(double) * Assembly.getNumElements(), cudaMemcpyDeviceToHost) );

        file_index++;
        ss.str( string() );
        ss.clear();
        ss << "vtk/tdo";
        ss << file_index;
        ss << fileformat;
        WriteVectorToVTK(chi, "chi", ss.str(), dim, Assembly.getGridSize(), h, Assembly.getNumElements() );
    }

    cudaDeviceSynchronize();

    }

    cudaDeviceSynchronize();
}

    // PTAP_GPU consider using 2d blocks? :
    // https://www.quantstart.com/articles/Matrix-Matrix-Multiplication-on-the-GPU-with-Nvidia-CUDA/

// print_GPU<<<1,1>>> ( d_res0 );