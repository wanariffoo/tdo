
#include <iostream>
#include "assemble.h"
#include <cmath>
#include "cudakernels.h"
#include "tdo.h"

using namespace std;


TDO::TDO(double* d_u, double* d_kai, double h, size_t dim, double beta, double eta, size_t numElements, size_t num_rows, double* d_A_local, vector<size_t*> d_node_index)
 : m_d_u(d_u), m_d_kai(d_kai), m_beta(beta), m_h(h), m_dim(dim), m_eta(eta), m_numElements(numElements), m_num_rows(num_rows), m_d_A_local(d_A_local), m_d_node_index(d_node_index)
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

    CUDA_CALL( cudaMalloc( (void**)&d_df, sizeof(double) * m_numElements ) );
    CUDA_CALL( cudaMemset(d_df, 0, sizeof(double) * m_numElements) );

    CUDA_CALL( cudaMalloc( (void**)&m_d_temp, sizeof(double) * m_num_rows) );
    CUDA_CALL( cudaMemset(m_d_temp, 0, sizeof(double) * m_num_rows) );

    return true;

}

bool TDO::innerloop()
{

    // calculating the driving force of each element
    // df[] = ( 1 / 2*omega ) * ( p * pow(kai[], p - 1 ) ) * sum( u^T * A_local * u * det(J) )

    // TODO: there's no jacobi here, assumed det_J = 1 for now I think
    for ( int i = 0 ; i < m_numElements ; i++ )
    calcDrivingForce ( &d_df[i], &m_d_kai[i], 3, m_d_temp, m_d_u, m_d_node_index[i], m_d_A_local, m_num_rows, m_local_volume, m_gridDim, m_blockDim );
    // uTAu_GPU(m_d_uTAu, m_d_u, size_t *node_index, double* value, size_t* index, size_t max_row_size, size_t num_rows);

    // printVector_GPU<<<1,4>>> ( m_d_temp, 4 );
    // printVector_GPU<<<1,4>>> ( m_d_kai, 4 );
    printVector_GPU<<<1,4>>> ( d_df, 4 );
    cudaDeviceSynchronize();
    // ( 1 / 2*element_volume ) * p * pow(kai_element, (p-1) ) * u^T * element_stiffness_matrix * u
    // UpdateDrivingForce<<<1,numElements>>> ( d_df, d_uTKu, 3, d_kai, l_volume, numElements);
    return true;

}