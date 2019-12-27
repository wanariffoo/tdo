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
    // domain dimensions
    size_t dim = 2;
    size_t Nx = 1;
    size_t Ny = 1;

    // number of levels in GMG
    size_t numLevels = 2;
    size_t topLev = numLevels - 1;

    // rows and cols of each grid
    vector<size_t> num_rows;
    vector<size_t> num_cols;
    num_rows.resize(numLevels);
    num_cols.resize(numLevels);
    
    // DOFs of each level
    for ( int i = 0 ; i < numLevels ; i++ )
        {
            num_rows[i] = calcDOF( Nx, Ny, dim );
            num_cols[i] = calcDOF( Nx, Ny, dim );
            Nx++;
            Ny++;
        }

    vector<size_t> max_row_size;
    max_row_size.resize(numLevels);
    

    // set DOF with boundary conditions
    vector<size_t> bc_index = {0, 1, 6, 7, 12, 13};



    // displacement vector
    vector<double> u(num_rows[topLev]);
    double* d_u;

    // force vector
    vector<double> b(num_rows[topLev]);
    double* d_b;

    // add forces
    b[5] = -10;

    // residuum vector
    double* d_r;

    // correction vector
    double* d_c;

    // function : assemble()
    // will result in d_value[numLevels], d_index[numLevels], max_row_size[numLevels]

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

    vector<double> value;
    vector<size_t> index;

    vector<double*> d_value;
    d_value.resize(3);
    vector<size_t*> d_index;
    d_index.resize(2);

    for ( int i = 0 ; i < bc_index.size() ; ++i )
        applyMatrixBC(&A_g[0], bc_index[i], num_rows[topLev], num_cols[topLev]);

        

    // get max row size
    max_row_size[topLev] = getMaxRowSize(A_g, num_rows[topLev], num_cols[topLev]); // TODO: might be not working
    max_row_size[0] = 4;    // TODO: need to figure how to get this number
    max_row_size[1] = 13;

    value.resize(max_row_size[topLev] * num_rows[topLev], 0.0);
    index.resize(max_row_size[topLev] * num_rows[topLev], 0);

    // TODO: something's not right with transform. The first element is skipped
    // transformToELL(A_g, &value[topLev], &index[topLev], max_row_size[topLev], num_rows[topLev]);

    value = { 1,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	1,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	-4066334.72,	185606.4,	13304204.8,	-4066334.72,	-185606.4,	-3325952,	2400153.6,	1480473.6,	-3325952,	-2400153.6,	0,	0,	0,	-185606.4,	740236.8,	13304204.8,	185606.4,	740236.8,	2400153.6,	-3325952,	-8132669.44,	-2400153.6,	-3325952,	0,	0,	0,	-4066334.72,	185606.4,	6652102.4,	-2400134.4,	-3325952,	2400153.6,	740236.8,	-185606.4,	0,	0,	0,	0,	0,	-185606.4,	740236.8,	-2400134.4,	6652102.4,	2400153.6,	-3325952,	185606.4,	-4066334.72,	0,	0,	0,	0,	0,	1,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	1,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	-3325952,	-2400153.6,	1480473.6,	-3325952,	2400153.6,	-8132669.44,	26608409.6,	-8132669.44,	-3325952,	2400153.6,	1480473.6,	-3325952,	-2400153.6,	-2400153.6,	-3325952,	-8132669.44,	2400153.6,	-3325952,	1480473.6,	26608409.6,	1480473.6,	2400153.6,	-3325952,	-8132669.44,	-2400153.6,	-3325952,	-3325952,	-2400153.6,	740236.8,	185606.4,	-8132669.44,	13304204.8,	-3325952,	2400153.6,	740236.8,	-185606.4,	0,	0,	0,	-2400153.6,	-3325952,	-185606.4,	-4066334.72,	1480473.6,	13304204.8,	2400153.6,	-3325952,	185606.4,	-4066334.72,	0,	0,	0,	1,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	1,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	-3325952,	-2400153.6,	1480473.6,	-3325952,	2400153.6,	-4066334.72,	-185606.4,	13304204.8,	-4066334.72,	185606.4,	0,	0,	0,	-2400153.6,	-3325952,	-8132669.44,	2400153.6,	-3325952,	185606.4,	740236.8,	13304204.8,	-185606.4,	740236.8,	0,	0,	0,	-3325952,	-2400153.6,	740236.8,	185606.4,	-4066334.72,	-185606.4,	6652102.4,	2400134.4,	0,	0,	0,	0,	0,	-2400153.6,	-3325952,	-185606.4,	-4066334.72,	185606.4,	740236.8,	2400134.4,	6652102.4,	0,	0,	0,	0,	0 };
    index = { 0,	18,	18,	18,	18,	18,	18,	18,	18,	18,	18,	18,	18,	1,	18,	18,	18,	18,	18,	18,	18,	18,	18,	18,	18,	18,	0,	1,	2,	4,	5,	6,	7,	8,	10,	11,	18,	18,	18,	0,	1,	3,	4,	5,	6,	7,	9,	10,	11,	18,	18,	18,	2,	3,	4,	5,	8,	9,	10,	11,	18,	18,	18,	18,	18,	2,	3,	4,	5,	8,	9,	10,	11,	18,	18,	18,	18,	18,	6,	18,	18,	18,	18,	18,	18,	18,	18,	18,	18,	18,	18,	7,	18,	18,	18,	18,	18,	18,	18,	18,	18,	18,	18,	18,	0,	1,	2,	4,	5,	6,	8,	10,	12,	13,	14,	16,	17,	0,	1,	3,	4,	5,	7,	9,	11,	12,	13,	15,	16,	17,	2,	3,	4,	5,	8,	10,	14,	15,	16,	17,	18,	18,	18,	2,	3,	4,	5,	9,	11,	14,	15,	16,	17,	18,	18,	18,	12,	18,	18,	18,	18,	18,	18,	18,	18,	18,	18,	18,	18,	13,	18,	18,	18,	18,	18,	18,	18,	18,	18,	18,	18,	18,	6,	7,	8,	10,	11,	12,	13,	14,	16,	17,	18,	18,	18,	6,	7,	9,	10,	11,	12,	13,	15,	16,	17,	18,	18,	18,	8,	9,	10,	11,	14,	15,	16,	17,	18,	18,	18,	18,	18,	8,	9,	10,	11,	14,	15,	16,	17,	18,	18,	18,	18,	18 };
        
    vector<double> value0 = {1,	0,	0,	0,	1,	0,	0,	0,	6640200,	-2400000,	735240,	-185610,	-2400000,	6640200,	185610,	-4075000,	1,	0,	0,	0,	1,	0,	0,	0,	735240,	185610,	6640200,	2400000,	-185610,	-4075000,	2400000,	6640200};
    vector<size_t> index0 = {0,	8,	8,	8,	1,	8,	8,	8,	2,	3,	6,	7,	2,	3,	6,	7,	4,	8,	8,	8,	5,	8,	8,	8,	2,	3,	6,	7,	2,	3,	6,	7};


    // prolongation matrix

    vector<double> p_value = { 1,	0, 1,	0, 0.5,	0, 0.5,	0, 1,	0, 1,	0, 0,	0, 0,	0, 0.25,	0.25, 0.25,	0.25, 0.5,	0.5, 0.5,	0.5, 1,	0, 1,	0, 0.5,	0, 0.5,	0, 1,	0, 0,	0};
    vector<size_t> p_index = { 0,	18, 1,	18, 2,	18, 3,	18, 2,	18, 3,	18, 18,	18, 18,	18, 2,	6, 3,	7, 2,	6, 3,	7, 4,	18, 5,	18, 6,	18, 7,	18, 6,	18, 7,	18};



    vector<double*> d_p_value;
    vector<size_t*> d_p_index;
    d_p_value.resize(numLevels - 1);
    d_p_index.resize(numLevels - 1);

    vector<size_t> p_max_row_size;
    p_max_row_size.resize(numLevels - 1);
    p_max_row_size[0] = 2;

    // cuda
    CUDA_CALL( cudaMalloc((void**)&d_u, sizeof(double) * num_rows[topLev]) );
    CUDA_CALL( cudaMalloc((void**)&d_b, sizeof(double) * num_rows[topLev]) );
    CUDA_CALL( cudaMalloc((void**)&d_r, sizeof(double) * num_rows[topLev]) );
    CUDA_CALL( cudaMalloc((void**)&d_c, sizeof(double) * num_rows[topLev]) );
    CUDA_CALL( cudaMalloc((void**)&d_value[0], sizeof(double) * max_row_size[0]*num_rows[0]) );
    CUDA_CALL( cudaMalloc((void**)&d_index[0], sizeof(size_t) * max_row_size[0]*num_rows[0]) );
    CUDA_CALL( cudaMalloc((void**)&d_value[1], sizeof(double) * max_row_size[topLev]*num_rows[topLev]) );
    CUDA_CALL( cudaMalloc((void**)&d_index[1], sizeof(size_t) * max_row_size[topLev]*num_rows[topLev]) );
    CUDA_CALL( cudaMalloc((void**)&d_p_value[0], sizeof(double) * 36 ) );
    CUDA_CALL( cudaMalloc((void**)&d_p_index[0], sizeof(size_t) * 36 ) );


    CUDA_CALL( cudaMemset(d_u, 0, sizeof(double) * num_rows[topLev]) );
    CUDA_CALL( cudaMemset(d_r, 0, sizeof(double) * num_rows[topLev]) );
    CUDA_CALL( cudaMemset(d_c, 0, sizeof(double) * num_rows[topLev]) );

    CUDA_CALL( cudaMemcpy(d_value[0], &value0[0], sizeof(double) * max_row_size[0]*num_rows[0], cudaMemcpyHostToDevice) );
    CUDA_CALL( cudaMemcpy(d_index[0], &index0[0], sizeof(size_t) * max_row_size[0]*num_rows[0], cudaMemcpyHostToDevice) );
    CUDA_CALL( cudaMemcpy(d_value[1], &value[0], sizeof(double) * max_row_size[topLev]*num_rows[topLev], cudaMemcpyHostToDevice) );
    CUDA_CALL( cudaMemcpy(d_index[1], &index[0], sizeof(size_t) * max_row_size[topLev]*num_rows[topLev], cudaMemcpyHostToDevice) );
    CUDA_CALL( cudaMemcpy(d_p_value[0], &p_value[0], sizeof(double) * 36, cudaMemcpyHostToDevice) );
    CUDA_CALL( cudaMemcpy(d_p_index[0], &p_index[0], sizeof(size_t) * 36, cudaMemcpyHostToDevice) );
    
    CUDA_CALL( cudaMemcpy(d_b, &b[0], sizeof(double) * num_rows[topLev], cudaMemcpyHostToDevice) );

    

    // NOTE: after assembly is done, you should have in the device:
    // d_value of each level
    // d_index of each level
    // max_row_size of each level
    // num_rows of each level
    // d_p_value of each level - 1
    // d_p_index of each level - 1
    // max_row_size of prol


    // /*
    // ##################################################################
    // #                           SOLVER                               #
    // ##################################################################
    // */


    Solver GMG(d_value, d_index, max_row_size, d_p_value, d_p_index, p_max_row_size, d_u, d_b, numLevels, num_rows, num_cols);

    GMG.init();
    GMG.set_num_prepostsmooth(1,1);

    cudaDeviceSynchronize();
    GMG.solve(d_u, d_b);

    cudaDeviceSynchronize();

    
}


// print_GPU<<<1,1>>> ( d_res0 );