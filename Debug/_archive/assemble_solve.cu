/*
    
*/

#include <iostream>
#include <cuda.h>
#include <vector>
#include <cuda_runtime.h>
// #include "../include/mycudaheader.h"
// #include "precond.h"
#include "cudakernels.h"

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
    
    // NOTE: temp
    
    // previous residuum
    double *d_res0;
    CUDA_CALL( cudaMalloc((void**)&d_res0, sizeof(double) * num_rows) );
    CUDA_CALL( cudaMemset(d_res0, 0, sizeof(double) * num_rows) );
    
    // current residuum
    double *d_res;
    CUDA_CALL( cudaMalloc((void**)&d_res, sizeof(double) * num_rows) );
    CUDA_CALL( cudaMemset(d_res, 0, sizeof(double) * num_rows) );

    // minimum required residuum for convergence
    double *d_m_minRes;
    CUDA_CALL( cudaMalloc((void**)&d_m_minRes, sizeof(double)) );
    CUDA_CALL( cudaMemset(d_m_minRes, 1.000e-99, sizeof(double)) );
    
    // minimum required reduction for convergence
    double *d_m_minRed;
    CUDA_CALL( cudaMalloc((void**)&d_m_minRed, sizeof(double)) );
    CUDA_CALL( cudaMemset(d_m_minRed, 1.000e-10, sizeof(double)) );
    
    // temporary residuum vectors for GMG
    vector<double*> d_rtmp;
    d_rtmp.resize(2);
    CUDA_CALL( cudaMalloc((void**)&d_rtmp[0], sizeof(double) * 8 ) );
    CUDA_CALL( cudaMemset(d_rtmp[0], 0, sizeof(double) * 8 ) );
    CUDA_CALL( cudaMalloc((void**)&d_rtmp[1], sizeof(double) * 18 ) );
    CUDA_CALL( cudaMemset(d_rtmp[1], 0, sizeof(double) * 18 ) );

    // temporary correction vectors for GMG
    vector<double*> d_ctmp;
    d_ctmp.resize(2);
    CUDA_CALL( cudaMalloc((void**)&d_ctmp[0], sizeof(double) * 8 ) );
    CUDA_CALL( cudaMemset(d_ctmp[0], 0, sizeof(double) * 8 ) );
    CUDA_CALL( cudaMalloc((void**)&d_ctmp[1], sizeof(double) * 18 ) );
    CUDA_CALL( cudaMemset(d_ctmp[1], 0, sizeof(double) * 18 ) );
    

    size_t m_numPreSmooth = 1;






    /*
    ##################################################################
    #                           SOLVER                               #
    ##################################################################
    */


    // r = b - A*u
    ComputeResiduum_GPU<<<gridDim,blockDim>>>(num_rows, max_row_size, d_value, d_index, d_u, d_r, d_b);

    // d_res0 = norm(d_r)
    norm_GPU(d_res0, d_r, num_rows, gridDim, blockDim);
        
    // res = res0;
    equals_GPU<<<1,1>>>(d_res, d_res0);	

    printInitialResult_GPU<<<1,1>>>(d_res0, d_m_minRes, d_m_minRed);

    
    
    // // foo loop

    // // DEBUG:
    // // precond(d_c, d_r);
    
    // std::size_t topLev = numLevels - 1;
    
	// // reset correction
    // setToZero<<<gridDim, blockDim>>>(d_c, num_rows);
    
    // // Vector<double> rtmp(r);
    // vectorEquals_GPU<<<gridDim, blockDim>>>( d_rtmp[topLev], d_r, num_rows );
    

    // // precond_add_update_GPU(d_gmg_c[topLev], d_rtmp[topLev], topLev, m_gamma);

    // // DEBUG:
    // // Level below
	// 	dim3 blockDim_;
	// 	dim3 gridDim_;
	// 	calculateDimensions(8, blockDim_, gridDim_);
		
	// 	// Level below
	// 	dim3 blockDim_cols;
	// 	dim3 gridDim_cols;
	// 	calculateDimensions(8, blockDim_cols, gridDim_cols);
    
    // // NOTE: no need this I think
    // // setToZero<<< gridDim, blockDim >>>( d_ctmp[lev], 18 );		
    
    // Jacobi_Precond_GPU<<<gridDim,blockDim>>>(d_c, d_value, d_rtmp[1], 18);





    cudaDeviceSynchronize();
}


// print_GPU<<<1,1>>> ( d_res0 );