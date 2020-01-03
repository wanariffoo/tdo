
#include <iostream>
#include "assemble.h"
#include <cmath>
#include "cudakernels.h"
#include "tdo.h"

using namespace std;


TDO::TDO(double* d_u, double* d_kai, double h, size_t dim, double beta, double eta, size_t numElements, double* d_A_local)
 : m_d_u(d_u), m_d_kai(d_kai), m_beta(beta), m_h(h), m_dim(dim), m_eta(eta), m_numElements(numElements), m_d_A_local(d_A_local)
{
    // inner loop frequency, n
    m_n = (6 / m_eta) * ( m_beta / (m_h*m_h) );

    // local volume
    m_local_volume = pow(m_h, m_dim);

}

bool TDO::init()
{

    calculateDimensions(m_numElements, m_gridDim, m_blockDim);

        
    cudaMalloc( (void**)&m_d_uTAu, sizeof(double) * m_numElements);
    cudaMemset( m_d_uTAu, 0, sizeof(double) * m_numElements);



    return true;

}

bool TDO::innerloop()
{
    size_t num_rows = 8;
    // DEBUG: TEST
    m_d_node_index.resize(4);
    vector<size_t> node_index0 = {0, 1, 3, 4};
    vector<size_t> node_index1 = {1, 2, 4, 5};
    vector<size_t> node_index2 = {3, 4, 6, 7};
    vector<size_t> node_index3 = {4, 5, 7, 8};

    vector<double> temp(num_rows, 0.0);
    CUDA_CALL ( cudaMalloc( (void**)&m_d_temp, sizeof(double) * num_rows) );
    CUDA_CALL ( cudaMemcpy( m_d_temp, &temp[0], sizeof(double) * num_rows, cudaMemcpyHostToDevice) );

    vector<double> df = {0, 0, 0, 0};
    double* d_df;
    CUDA_CALL( cudaMalloc( (void**)&d_df, sizeof(double) * 4 ) );
    CUDA_CALL( cudaMemcpy(d_df, &df[0], sizeof(double) * 4, cudaMemcpyHostToDevice) );  

    CUDA_CALL( cudaMalloc( (void**)&m_d_node_index[0], sizeof(size_t) * 4 ) );
    CUDA_CALL( cudaMemcpy(m_d_node_index[0], &node_index0[0], sizeof(size_t) * 4, cudaMemcpyHostToDevice) );  

    CUDA_CALL( cudaMalloc( (void**)&m_d_node_index[1], sizeof(size_t) * 4 ) );
    CUDA_CALL( cudaMemcpy(m_d_node_index[1], &node_index1[0], sizeof(size_t) * 4, cudaMemcpyHostToDevice) );  

    CUDA_CALL( cudaMalloc( (void**)&m_d_node_index[2], sizeof(size_t) * 4 ) );
    CUDA_CALL( cudaMemcpy(m_d_node_index[2], &node_index2[0], sizeof(size_t) * 4, cudaMemcpyHostToDevice) );  

    CUDA_CALL( cudaMalloc( (void**)&m_d_node_index[3], sizeof(size_t) * 4 ) );
    CUDA_CALL( cudaMemcpy(m_d_node_index[3], &node_index3[0], sizeof(size_t) * 4, cudaMemcpyHostToDevice) );  

    for ( int i = 0 ; i < m_numElements ; i++ )
    calcDrivingForce ( &d_df[i], &m_d_kai[i], 3, m_d_temp, m_d_u, m_d_node_index[i], m_d_A_local, num_rows, m_gridDim, m_blockDim );
    // uTAu_GPU(m_d_uTAu, m_d_u, size_t *node_index, double* value, size_t* index, size_t max_row_size, size_t num_rows);

    // printVector_GPU<<<1,4>>> ( m_d_temp, 4 );
    // printVector_GPU<<<1,4>>> ( m_d_kai, 4 );
    printVector_GPU<<<1,4>>> ( d_df, 4 );
    cudaDeviceSynchronize();
    // ( 1 / 2*element_volume ) * p * pow(kai_element, (p-1) ) * u^T * element_stiffness_matrix * u
    // UpdateDrivingForce<<<1,numElements>>> ( d_df, d_uTKu, 3, d_kai, l_volume, numElements);
    return true;

}