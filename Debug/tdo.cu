
#include <iostream>
#include "assemble.h"
#include <cmath>
#include "cudakernels.h"
#include "tdo.h"

using namespace std;


TDO::TDO(double* d_u, double* d_kai, double h, size_t dim, double beta, double eta, size_t numElements, size_t num_rows, double* d_A_local, vector<size_t*> d_node_index, vector<size_t> N, double del_t)
 : m_d_u(d_u), m_d_kai(d_kai), m_beta(beta), m_h(h), m_dim(dim), m_eta(eta), m_numElements(numElements), m_num_rows(num_rows), m_d_A_local(d_A_local), m_d_node_index(d_node_index), m_N(N), m_del_t(del_t)
{
    // inner loop frequency, n
    m_n = (6 / m_eta) * ( m_beta / (m_h*m_h) );

    // local volume
    m_local_volume = pow(m_h, m_dim);

    
}

bool TDO::init()
{
    

    calculateDimensions(m_numElements, m_gridDim, m_blockDim);

        
    CUDA_CALL( cudaMalloc( (void**)&m_d_df, sizeof(double) * m_numElements ) );
    CUDA_CALL( cudaMemset(m_d_df, 0, sizeof(double) * m_numElements) );

    CUDA_CALL( cudaMalloc( (void**)&m_d_temp, sizeof(double) * m_num_rows) );
    CUDA_CALL( cudaMemset(m_d_temp, 0, sizeof(double) * m_num_rows) );

    CUDA_CALL( cudaMalloc( (void**)&m_d_mutex, sizeof(int) ) );

    CUDA_CALL( cudaMalloc( (void**)&m_d_lambda_tr, sizeof(double) ) );
    CUDA_CALL( cudaMalloc( (void**)&m_d_lambda_l, sizeof(double) ) );
    CUDA_CALL( cudaMalloc( (void**)&m_d_lambda_u, sizeof(double) ) );
    CUDA_CALL( cudaMalloc( (void**)&m_d_kai_tr, sizeof(double) * m_numElements) );
    CUDA_CALL( cudaMalloc( (void**)&m_d_rho_tr, sizeof(double) ) );

    CUDA_CALL( cudaMemset( m_d_lambda_l, 0, sizeof(double) ) );
    CUDA_CALL( cudaMemset( m_d_lambda_tr, 0, sizeof(double) ) );
    CUDA_CALL( cudaMemset( m_d_lambda_u, 0, sizeof(double) ) );
    CUDA_CALL( cudaMemset( m_d_kai_tr, 0, sizeof(double) * m_numElements) );
    CUDA_CALL( cudaMemset( m_d_rho_tr, 0, sizeof(double) ) );

    return true;
}

bool TDO::innerloop()
{

    // calculating the driving force of each element
    // df[] = ( 1 / 2*omega ) * ( p * pow(kai[], p - 1 ) ) * sum( u^T * A_local * u * det(J) )

    // TODO: there's no jacobi here, assumed det_J = 1 for now I think
    // CHECK: jacobi maybe is already in A local??
    // CHECK: here temp[] array or scalar?



    for ( int i = 0 ; i < m_numElements ; i++ )
        calcDrivingForce ( &m_d_df[i], &m_d_kai[i], 3, m_d_temp, m_d_u, m_d_node_index[i], m_d_A_local, m_num_rows, m_gridDim, m_blockDim );


    // d_temp = u^T * A * u
    vectorEquals_GPU<<<m_gridDim,m_blockDim>>>(m_d_temp, m_d_df, m_numElements);


    // NOTE:
    //// for loop
     for ( int j = 0 ; j < m_n; j++ )
    {

        // df[] = ( 1 / 2*element_volume ) * p * pow(kai_element, (p-1) ) * temp[]
        // temp[] = u[]^T * A * u[]
        UpdateDrivingForce<<<m_gridDim,m_blockDim>>> ( m_d_df, m_d_temp, 3, m_d_kai, m_local_volume, m_numElements );

        // cudaDeviceSynchronize();
        // printVector_GPU<<<1,4>>> ( m_d_df, 4 );

        // bisection algo: 
            
        setToZero<<<1,1>>>(m_d_lambda_tr, 1);
        calcLambdaUpper<<< m_gridDim, m_blockDim >>> (m_d_df, m_d_lambda_u, m_d_mutex, m_beta, m_d_kai, m_eta, m_N[0], m_numElements);
        calcLambdaLower<<< m_gridDim, m_blockDim >>> (m_d_df, m_d_lambda_l, m_d_mutex, m_beta, m_d_kai, m_eta, m_N[0], m_numElements);
        

        for ( int i = 0 ; i < 30 ; i++ )
        {
            calcKaiTrial<<<m_gridDim,m_blockDim>>> ( m_d_kai, m_d_df, m_d_lambda_tr, m_del_t, m_eta, m_beta, m_d_kai_tr, m_N[0], m_numElements);
            setToZero<<<1,1>>>(m_d_rho_tr, 1);
            sumOfVector_GPU <<< m_gridDim, m_blockDim >>> (m_d_rho_tr, m_d_kai_tr, m_numElements);

            calcLambdaTrial<<<1,1>>>( m_d_rho_tr, 0.4, m_d_lambda_tr, m_d_lambda_l, m_d_lambda_u);
        }

        // kai(j) = kai(j+1)
        vectorEquals_GPU<<<m_gridDim,m_blockDim>>>( m_d_kai, m_d_kai_tr, m_numElements );

    }

    printVector_GPU<<<m_gridDim,m_blockDim>>>( m_d_kai, m_numElements);
    cudaDeviceSynchronize();

    cudaDeviceSynchronize();
    return true;

}