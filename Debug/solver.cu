#include <iostream>
#include "solver.h"
#include "cudakernels.h"
using namespace std;

Solver::Solver(double* d_value, size_t* d_index, size_t max_row_size, double* d_u, double* d_b, size_t numLevels, size_t num_rows, size_t num_cols) 
    : m_d_value(d_value), m_d_index(d_index), m_max_row_size(max_row_size), m_numLevels(numLevels), m_num_rows(num_rows), m_num_cols(num_cols)
{
    cout << "constructor" << endl;

    
}

// TODO:
Solver::~Solver()
{
    CUDA_CALL( cudaFree(d_res0) );
    CUDA_CALL( cudaFree(d_res) );
    CUDA_CALL( cudaFree(d_m_minRes) );
    CUDA_CALL( cudaFree(d_m_minRed) );

    for ( int i = 0 ; i < m_numLevels ; i++ )
    {
        CUDA_CALL( cudaFree(d_rtmp[i]) );
        CUDA_CALL( cudaFree(d_ctmp[i]) );
    }
}

bool Solver::init()
{
        cout << "init" << endl;

        
        
        // calculate cuda grid and block dimensions of each level

        m_gridDim.resize(m_numLevels);
        m_blockDim.resize(m_numLevels);
        // TODO: for loop
        calculateDimensions(8, m_gridDim[0], m_blockDim[0]);
        calculateDimensions(18, m_gridDim[1], m_blockDim[1]);


        CUDA_CALL( cudaMalloc((void**)&m_d_r, sizeof(double) * m_num_rows) );
        CUDA_CALL( cudaMemset(m_d_r, 0, sizeof(double) * m_num_rows) );
        CUDA_CALL( cudaMalloc((void**)&m_d_c, sizeof(double) * m_num_rows) );
        CUDA_CALL( cudaMemset(m_d_c, 0, sizeof(double) * m_num_rows) );


        // previous residuum
        CUDA_CALL( cudaMalloc((void**)&d_res0, sizeof(double)) );
        CUDA_CALL( cudaMemset(d_res0, 0, sizeof(double)) );
        
        // current residuum
        CUDA_CALL( cudaMalloc((void**)&d_res, sizeof(double)) );
        CUDA_CALL( cudaMemset(d_res, 0, sizeof(double)) );
    
        // minimum required residuum for convergence
        // d_m_minRes;
        CUDA_CALL( cudaMalloc((void**)&d_m_minRes, sizeof(double)) );
        CUDA_CALL( cudaMemset(d_m_minRes, 1.000e-99, sizeof(double)) );
        
        // minimum required reduction for convergence
        // d_m_minRed;
        CUDA_CALL( cudaMalloc((void**)&d_m_minRed, sizeof(double)) );
        CUDA_CALL( cudaMemset(d_m_minRed, 1.000e-10, sizeof(double)) );
        
        // temporary residuum vectors for GMG
        d_rtmp.resize(m_numLevels);
        CUDA_CALL( cudaMalloc((void**)&d_rtmp[0], sizeof(double) * 8 ) );
        CUDA_CALL( cudaMemset(d_rtmp[0], 0, sizeof(double) * 8 ) );
        CUDA_CALL( cudaMalloc((void**)&d_rtmp[1], sizeof(double) * 18 ) );
        CUDA_CALL( cudaMemset(d_rtmp[1], 0, sizeof(double) * 18 ) );
    
        // temporary correction vectors for GMG
        d_ctmp.resize(m_numLevels);
        CUDA_CALL( cudaMalloc((void**)&d_ctmp[0], sizeof(double) * 8 ) );
        CUDA_CALL( cudaMemset(d_ctmp[0], 0, sizeof(double) * 8 ) );
        CUDA_CALL( cudaMalloc((void**)&d_ctmp[1], sizeof(double) * 18 ) );
        CUDA_CALL( cudaMemset(d_ctmp[1], 0, sizeof(double) * 18 ) );

    return true;
}

void Solver::set_num_presmooth(size_t n)
{
    m_numPreSmooth = n;
}

void Solver::set_num_postsmooth(size_t n)
{
    m_numPostSmooth = n;
}

bool Solver::solve(double* d_u, double* d_b)
{
    cout << "solve" << endl;

    // r = b - A*u
    ComputeResiduum_GPU<<<m_gridDim[1], m_blockDim[1]>>>(m_num_rows, m_max_row_size, m_d_value, m_d_index, d_u, m_d_r, d_b);
    
    // d_res0 = norm(m_d_r)
    norm_GPU(d_res0, m_d_r, m_num_rows, m_gridDim[1], m_blockDim[1]);
    
    // res = res0;
    equals_GPU<<<1,1>>>(d_res, d_res0);	

    
    cout << endl;
    cout << "## Iterative solver ############################################################" << endl;
    cout << "  Iter     Residuum       Required       Rate        Reduction     Required" << endl;
    printInitialResult_GPU<<<1,1>>>(d_res0, d_m_minRes, d_m_minRed);
        
    return true;
}

// cudaDeviceSynchronize();
// print_GPU<<<1,1>>>( d_res0 );