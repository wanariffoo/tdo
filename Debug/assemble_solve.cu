/*
    
*/

#include <iostream>
#include <cuda.h>
#include <vector>
#include <cuda_runtime.h>
// #include "../include/mycudaheader.h"
// #include "precond.h"
#include "cudakernels.h"
#include "solver.h"

using namespace std;


int main()
{
    size_t num_rows = 18;
    size_t num_cols = 18;

    size_t max_row_size;
    size_t N = 2;

    vector<size_t> bc_index = {0, 1, 6, 7, 12, 13};

    // number of levels in GMG
    size_t numLevels = 2;


    // displacement vector
    vector<double> u(num_rows);
    double* d_u;

    // force vector
    vector<double> b(num_rows);
    double* d_b;

    // add forces
    b[5] = -10;

    // residuum vector
    double* d_r;

    // correction vector
    double* d_c;

    vector<double> A_g = {
        6652102.4,	2400134.4,	-4066334.72,	-185606.4,	0,	0,	740236.8,	185606.4,	-3325952,	-2400153.6,	0,	0,	0,	0,	0,	0,	0,	0,
        2400134.4,	6652102.4,	185606.4,	740236.8,	0,	0,	-185606.4,	-4066334.72,	-2400153.6,	-3325952,	0,	0,	0,	0,	0,	0,	0,	0,
        -4066334.72,	185606.4,	13304204.8,	0,	-4066334.72,	-185606.4,	-3325952,	2400153.6,	1480473.6,	0,	-3325952,	-2400153.6,	0,	0,	0,	0,	0,	0,
        -185606.4,	740236.8,	0,	13304204.8,	185606.4,	740236.8,	2400153.6,	-3325952,	0,	-8132669.44,	-2400153.6,	-3325952,	0,	0,	0,	0,	0,	0,
        0,	0,	-4066334.72,	185606.4,	6652102.4,	-2400134.4,	0,	0,	-3325952,	2400153.6,	740236.8,	-185606.4,	0,	0,	0,	0,	0,	0,
        0,	0,	-185606.4,	740236.8,	-2400134.4,	6652102.4,	0,	0,	2400153.6,	-3325952,	185606.4,	-4066334.72,	0,	0,	0,	0,	0,	0,
        740236.8,	-185606.4,	-3325952,	2400153.6,	0,	0,	13304204.8,	0,	-8132669.44,	0,	0,	0,	740236.8,	185606.4,	-3325952,	-2400153.6,	0,	0,
        185606.4,	-4066334.72,	2400153.6,	-3325952,	0,	0,	0,	13304204.8,	0,	1480473.6,	0,	0,	-185606.4,	-4066334.72,	-2400153.6,	-3325952,	0,	0,
        -3325952,	-2400153.6,	1480473.6,	0,	-3325952,	2400153.6,	-8132669.44,	0,	26608409.6,	0,	-8132669.44,	0,	-3325952,	2400153.6,	1480473.6,	0,	-3325952,	-2400153.6,
        -2400153.6,	-3325952,	0,	-8132669.44,	2400153.6,	-3325952,	0,	1480473.6,	0,	26608409.6,	0,	1480473.6,	2400153.6,	-3325952,	0,	-8132669.44,	-2400153.6,	-3325952,
        0,	0,	-3325952,	-2400153.6,	740236.8,	185606.4,	0,	0,	-8132669.44,	0,	13304204.8,	0,	0,	0,	-3325952,	2400153.6,	740236.8,	-185606.4,
        0,	0,	-2400153.6,	-3325952,	-185606.4,	-4066334.72,	0,	0,	0,	1480473.6,	0,	13304204.8,	0,	0,	2400153.6,	-3325952,	185606.4,	-4066334.72,
        0,	0,	0,	0,	0,	0,	740236.8,	-185606.4,	-3325952,	2400153.6,	0,	0,	6652102.4,	-2400134.4,	-4066334.72,	185606.4,	0,	0,
        0,	0,	0,	0,	0,	0,	185606.4,	-4066334.72,	2400153.6,	-3325952,	0,	0,	-2400134.4,	6652102.4,	-185606.4,	740236.8,	0,	0,
        0,	0,	0,	0,	0,	0,	-3325952,	-2400153.6,	1480473.6,	0,	-3325952,	2400153.6,	-4066334.72,	-185606.4,	13304204.8,	0,	-4066334.72,	185606.4,
        0,	0,	0,	0,	0,	0,	-2400153.6,	-3325952,	0,	-8132669.44,	2400153.6,	-3325952,	185606.4,	740236.8,	0,	13304204.8,	-185606.4,	740236.8,
        0,	0,	0,	0,	0,	0,	0,	0,	-3325952,	-2400153.6,	740236.8,	185606.4,	0,	0,	-4066334.72,	-185606.4,	6652102.4,	2400134.4,
        0,	0,	0,	0,	0,	0,	0,	0,	-2400153.6,	-3325952,	-185606.4,	-4066334.72,	0,	0,	185606.4,	740236.8,	2400134.4,	6652102.4
    };

    std::vector<double> value;
    std::vector<std::size_t> index;

    double *d_value;
    size_t *d_index;

    for ( int i = 0 ; i < bc_index.size() ; ++i )
        applyMatrixBC(&A_g[0], bc_index[i], num_rows, num_cols);

    // get max row size
    max_row_size = getMaxRowSize(A_g, num_rows, num_cols);

    value.resize(max_row_size*num_rows, 0.0);
    index.resize(max_row_size*num_rows, 0);

    transformToELL(A_g, value, index, max_row_size, num_rows);

    // cuda
    CUDA_CALL( cudaMalloc((void**)&d_u, sizeof(double) * num_rows) );
    CUDA_CALL( cudaMalloc((void**)&d_b, sizeof(double) * num_rows) );
    CUDA_CALL( cudaMalloc((void**)&d_r, sizeof(double) * num_rows) );
    CUDA_CALL( cudaMalloc((void**)&d_c, sizeof(double) * num_rows) );
    CUDA_CALL( cudaMalloc((void**)&d_value, sizeof(double) * max_row_size*num_rows) );
    CUDA_CALL( cudaMalloc((void**)&d_index, sizeof(size_t) * max_row_size*num_rows) );

    CUDA_CALL( cudaMemset(d_u, 0, sizeof(double) * num_rows) );
    CUDA_CALL( cudaMemset(d_r, 0, sizeof(double) * num_rows) );
    CUDA_CALL( cudaMemset(d_c, 0, sizeof(double) * num_rows) );
    

    CUDA_CALL( cudaMemcpy(d_value, &value[0], sizeof(double) * max_row_size*num_rows, cudaMemcpyHostToDevice) );
    CUDA_CALL( cudaMemcpy(d_index, &index[0], sizeof(size_t) * max_row_size*num_rows, cudaMemcpyHostToDevice) );
    CUDA_CALL( cudaMemcpy(d_b, &b[0], sizeof(double) * num_rows, cudaMemcpyHostToDevice) );

    dim3 gridDim;
    dim3 blockDim;
    
    // Calculating the required CUDA grid and block dimensions
    calculateDimensions(num_rows, gridDim, blockDim);


    /*
    ##################################################################
    #                           SOLVER                               #
    ##################################################################
    */


    Solver GMG(d_value, d_index, max_row_size, d_u, d_b, 2, num_rows, num_cols);

    GMG.init();
    GMG.set_num_presmooth(3);
    GMG.set_num_postsmooth(3);

    cudaDeviceSynchronize();
    GMG.solve(d_u, d_b);

    cudaDeviceSynchronize();

    
}


// print_GPU<<<1,1>>> ( d_res0 );