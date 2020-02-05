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
// TODO: enum : bc case 
    

int main()
{
  
    // create vtk files
    bool writeToVTK = true;

    // material properties
    double youngMod = 210e9;
    double poisson = 0.3;

    //// model set-up
    size_t numLevels = 2;
    
    vector<size_t> N;
    vector<vector<size_t>> bc_index(numLevels);
    // domain dimensions (x,y,z) on coarsest grid
    N = {1,1,1};

    // local element mesh size on coarsest grid
    double h_coarse = 1;





    size_t dim = N.size();
    bc_index = applyBC(N, numLevels, 0, dim);
    
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

    cout << "### GPU-accelerated Thermodynamic Topology Optimization ###" << endl;
    cout << "Number of Multigrid Levels: " << numLevels << endl;
    cout << "Dimension: " << dim << endl;
    cout << "Coarsest grid size = { " << N[0];
        for ( int i = 1 ; i < dim ; ++i )
            cout << ", " << N[i];
    cout << " }\n";
    cout << "Top-level mesh size = " << h << endl;

    /*
    ##################################################################
    #                           ASSEMBLY                             #
    ##################################################################
    */
    
    

    Assembler Assembly(dim, h, N, youngMod, poisson, rho, p, numLevels);
    Assembly.setBC(bc_index);
    Assembly.init(d_A_local, d_value, d_index, d_p_value, d_p_index, d_r_value, d_r_index, d_chi, num_rows, max_row_size, p_max_row_size, r_max_row_size, d_node_index);
    
    // cout << "Top-level number of rows = " << num_rows[numLevels - 1] << endl;
    // cout << "Assembly ... DONE" << endl;
    
    // // printELL_GPU<<<1,1>>> ( d_value[2], d_index[2], max_row_size[2], num_rows[2], num_rows[2]);
    // // printVector_GPU<<<1,10>>>( d_value[2], 10);

    // // vector u, b
    // vector<double> b(num_rows[numLevels - 1], 0);
    // double force = -1;

    // applyLoad(b, N, numLevels, 0, dim, force);




    // double* d_u;
    // double* d_b;

    // CUDA_CALL( cudaMalloc((void**)&d_u, sizeof(double) * num_rows[numLevels - 1] ) );
    // CUDA_CALL( cudaMalloc((void**)&d_b, sizeof(double) * num_rows[numLevels - 1] ) );

    // CUDA_CALL( cudaMemset(d_u, 0, sizeof(double) * num_rows[numLevels - 1]) );
    // CUDA_CALL( cudaMemcpy(d_b, &b[0], sizeof(double) * num_rows[numLevels - 1], cudaMemcpyHostToDevice) );

    // bar<<<1,1>>>(0,0,d_value[2], d_index[2], max_row_size[2]);

    // DEBUG:CHECK:
    // printELL_GPU<<<1,1>>> ( d_value[0], d_index[0], max_row_size[0], num_rows[0], num_rows[0]);
    // printELL_GPU<<<1,1>>> ( d_value[1], d_index[1], max_row_size[1], num_rows[1], num_rows[1]);
    // printELL_GPU<<<1,1>>> ( d_value[2], d_index[2], max_row_size[2], num_rows[2], num_rows[2]);
    // printELL_GPU<<<1,1>>> ( d_value[3], d_index[3], max_row_size[3], num_rows[3], num_rows[3]);

    // printELL_GPU<<<1,1>>> ( d_p_value[1], d_p_index[1], p_max_row_size[1], num_rows[2], num_rows[1]);




    // /*
    // ##################################################################
    // #                           SOLVER                               #
    // ##################################################################
    // */

    // Solver GMG(d_value, d_index, d_p_value, d_p_index, numLevels, num_rows, max_row_size, p_max_row_size, damp);
    // GMG.set_convergence_params(500, 1e-99, 1e-10);
    // // GMG.set_bs_convergence_params(10, 1e-99, 1e-10); // TODO: KIV
    

    // GMG.init();
    // GMG.set_verbose(1, 0);
    // GMG.set_num_prepostsmooth(3,3);
    // GMG.set_cycle('V');
    // // GMG.set_steps(15, 5); 
    
    // GMG.solve(d_u, d_b, d_value);
    // cudaDeviceSynchronize();

    // // // DEBUG:CHECK:
    // // // printVector_GPU<<<1,Assembly.getNumElements()>>>( d_chi, Assembly.getNumElements());
    // // printVector_GPU<<<1,num_rows[numLevels - 1]>>>( d_u, num_rows[numLevels - 1]);
    
    
    // // printELLrow(1, d_p_value[1], d_p_index[1], p_max_row_size[1], num_rows[2], num_rows[1]);

    




    // /*
    // ##################################################################
    // #                           TDO                                  #
    // ##################################################################
    // */

    
    // TDO tdo(d_u, d_chi, h, dim, betastar, etastar, Assembly.getNumElements(), local_num_rows, d_A_local, d_node_index, Assembly.getGridSize(), rho, numLevels, p);
    // tdo.init();
    // tdo.innerloop(d_u, d_chi);    // get updated d_chi

    // // // DEBUG:CHECK:
    // // printVector_GPU<<<1,Assembly.getNumElements()>>>( d_chi, Assembly.getNumElements());
    // // printVector_GPU<<<1,num_rows[numLevels - 1]>>>( d_u, num_rows[numLevels - 1]);

    
    // // TODO: write a function for this to make it neater
    // // vtk stuff
    // vector<double> chi(Assembly.getNumElements(), rho);
    // vector<double> u(Assembly.getNumNodes() * dim, 0);
    // string fileformat(".vtk");
    // int file_index = 0;
    // stringstream ss; 
    // ss << "vtk/tdo";
    // ss << file_index;
    // ss << fileformat;

    // if ( writeToVTK )
    // {
    //     WriteVectorToVTK(chi, u, ss.str(), dim, Assembly.getNumNodesPerDim(), h, Assembly.getNumElements(), Assembly.getNumNodes() );
        
    //     CUDA_CALL( cudaMemcpy(&chi[0], d_chi, sizeof(double) * Assembly.getNumElements(), cudaMemcpyDeviceToHost) );
    //     CUDA_CALL( cudaMemcpy(&u[0], d_u, sizeof(double) * u.size(), cudaMemcpyDeviceToHost) );

    //     file_index++;
    //     ss.str( string() );
    //     ss.clear();
    //     ss << "vtk/tdo";
    //     ss << file_index;
    //     ss << fileformat;
        
    //     WriteVectorToVTK(chi, u, ss.str(), dim, Assembly.getNumNodesPerDim(), h, Assembly.getNumElements(), Assembly.getNumNodes() );
    // }




//     // // // cudaDeviceSynchronize();
//     // // // printELL_GPU<<<1,1>>> ( d_value[0], d_index[0], max_row_size[0], num_rows[0], num_rows[0]);
//     // // // printELL_GPU<<<1,1>>> ( d_value[1], d_index[1], max_row_size[1], num_rows[1], num_rows[1]);
//     // printELL_GPU<<<1,1>>> ( d_value[2], d_index[2], max_row_size[2], num_rows[2], num_rows[2]);
//     // printELL_GPU<<<1,1>>> ( d_value[3], d_index[3], max_row_size[3], num_rows[3], num_rows[3]);
//     // // // cudaDeviceSynchronize();
//     // // // DEBUG:
//     // // // printVector_GPU<<<1,Assembly.getNumElements()>>>( d_chi, Assembly.getNumElements());

    

//     // // TODO: no need for R-matrix
//     // Assembly.UpdateGlobalStiffness(d_chi, d_value, d_index, d_p_value, d_p_index, d_r_value, d_r_index, d_A_local);
//     // cudaDeviceSynchronize();    


//     // printELLrow(2, d_value[2], d_index[2], max_row_size[2], num_rows[2], num_rows[2]);
//     // printELLrow(3, d_value[3], d_index[3], max_row_size[3], num_rows[3], num_rows[3]);

// //    for ( size_t i = 325 ; i < num_rows[3] ; i++ )
// //     {
// //         printELLrow_GPU<<<1,1>>> (i, d_value[3], d_index[3], max_row_size[3], num_rows[3], num_rows[3]);
// //         cudaDeviceSynchronize();    
// //     }

//     // // // DEBUG:
//     // // // // cudaDeviceSynchronize();
//     // // // printVector_GPU<<<1,num_rows[numLevels - 1]>>>( d_u, num_rows[numLevels - 1]);
//     // // // printVector_GPU<<<1,Assembly.getNumElements()>>>( d_chi, Assembly.getNumElements());

//     // // // A matrix
//     // // printELL_GPU<<<1,1>>> ( d_value[0], d_index[0], max_row_size[0], num_rows[0], num_rows[0]);
//     // // printELL_GPU<<<1,1>>> ( d_value[1], d_index[1], max_row_size[1], num_rows[1], num_rows[1]);

//     // int ELLlev = 2;
//     // for ( int i = 0 ; i < num_rows[ELLlev] ; i++ )
//     // {
//     //     printELLrow_GPU<<<1,1>>> (i, d_value[ELLlev], d_index[ELLlev], max_row_size[ELLlev], num_rows[ELLlev], num_rows[ELLlev]);
//     //     cudaDeviceSynchronize();    
//     // }

//     // printELL_GPU<<<1,1>>> ( d_value[3], d_index[3], max_row_size[3], num_rows[3], num_rows[3]);
//     // printELL_GPU<<<1,1>>> ( d_value[4], d_index[4], max_row_size[4], num_rows[4], num_rows[4]);

    

//     // printVector_GPU<<<2,1024>>>( d_value[2], max_row_size[2]*num_rows[2]);


//     // // // prolongation matrix
//     // printELL_GPU<<<1,1>>> ( d_p_value[0], d_p_index[0], p_max_row_size[0], num_rows[1], num_rows[0]);
//     // printELL_GPU<<<1,1>>> ( d_p_value[1], d_p_index[1], p_max_row_size[1], num_rows[2], num_rows[1]);
    
//     // // // restriction matrix
//     // // // printELL_GPU<<<1,1>>> ( d_r_value[0], d_r_index[0], r_max_row_size[0], num_rows[0], num_rows[1]);
//     // // // printELL_GPU<<<1,1>>> ( d_r_value[1], d_r_index[1], r_max_row_size[1], num_rows[1], num_rows[2]);


    ////////////////
    // ITERATION
    ////////////////
    

    // for ( int i = 1 ; i < 1 ; ++i )
    // {
    //     // TODO: something's wrong with the solver for N = {3,1}
    //     cout << "Calculating iteration " << i << " ... ";
    //     cudaDeviceSynchronize();
    //     GMG.reinit();
    //     GMG.set_verbose(0, 0);
    //     // GMG.set_convergence_params(1, 1e-99, 1e-10); // DEBUG:
    //     GMG.set_steps(50, 9); // NOTE: base solver has bugs, would only work when # iterations ~10
    //     GMG.solve(d_u, d_b, d_value);
    //     cudaDeviceSynchronize();

    //     // printVector_GPU<<<1,num_rows[numLevels - 1]>>>( d_u, num_rows[numLevels - 1]);

    //     // if (result)
    //         cout << "SUCCESS\n";


    //     tdo.innerloop(d_u, d_chi);
        
    //     // cudaDeviceSynchronize();
    //     // printVector_GPU<<<1,Assembly.getNumElements()>>>( d_chi, Assembly.getNumElements());
    //     // cout << "\n";

    //     if ( writeToVTK )
    //     { 
    //         CUDA_CALL( cudaMemcpy(&chi[0], d_chi, sizeof(double) * Assembly.getNumElements(), cudaMemcpyDeviceToHost) );
    //         CUDA_CALL( cudaMemcpy(&u[0], d_u, sizeof(double) * u.size(), cudaMemcpyDeviceToHost) );

    //         file_index++;
    //         ss.str( string() );
    //         ss.clear();
    //         ss << "vtk/tdo";
    //         ss << file_index;
    //         ss << fileformat;
            
    //         WriteVectorToVTK(chi, u, ss.str(), dim, Assembly.getNumNodesPerDim(), h, Assembly.getNumElements(), Assembly.getNumNodes() );

    //     }
            // cudaDeviceSynchronize();
    // }

    cudaDeviceSynchronize();
}

    // PTAP_GPU consider using 2d blocks? :
    // https://www.quantstart.com/articles/Matrix-Matrix-Multiplication-on-the-GPU-with-Nvidia-CUDA/

// print_GPU<<<1,1>>> ( d_res0 );