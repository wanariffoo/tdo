
#include <iostream>
#include "assemble.h"
#include <cmath>
#include "cudakernels.h"
#include "tdo.h"

using namespace std;


TDO::TDO(double* d_u, double* d_chi, double h, size_t dim, double betastar, double etastar, size_t numElements, size_t num_rows, double* d_A_local, vector<size_t*> d_node_index, vector<size_t> N, double rho, size_t numLevels, size_t p)
 : m_d_u(d_u), m_d_chi(d_chi), m_h(h), m_dim(dim), m_numElements(numElements), m_num_rows(num_rows), m_d_A_local(d_A_local), m_d_node_index(d_node_index), m_N(N), m_rho(rho), m_etastar(etastar), m_betastar(betastar), m_numLevels(numLevels), m_p(p)
{
    // inner loop frequency, n
    m_n = (6 / m_etastar) * ( m_betastar / (m_h*m_h) );
    m_del_t = 1.0 / m_n;

    
    // // TODO: calculate p_w
    // double g

    // TODO: reduction: calcP_w
    // calcP_w


    // TODO: betastar, etastar
    
    // TODO: del_t = 1.0 if 3D, see paper page 15
    

    // local volume
    // NOTE: wrong here because you thought m_h here is baselevel's, it's actually the finest level
    m_local_volume = pow(m_h, m_dim); 

    
    
}

bool TDO::init()
{

    calculateDimensions(m_numElements, m_gridDim, m_blockDim);

        
    CUDA_CALL( cudaMalloc( (void**)&m_d_df, sizeof(double) * m_numElements ) );
    CUDA_CALL( cudaMemset( m_d_df, 0, sizeof(double) * m_numElements) );

    CUDA_CALL( cudaMalloc( (void**)&m_d_uTAu, sizeof(double) * m_num_rows) );
    CUDA_CALL( cudaMemset( m_d_uTAu, 0, sizeof(double) * m_num_rows) );

    CUDA_CALL( cudaMalloc( (void**)&m_d_temp, sizeof(double) * m_num_rows) );
    CUDA_CALL( cudaMemset( m_d_temp, 0, sizeof(double) * m_num_rows) );

    CUDA_CALL( cudaMalloc( (void**)&m_d_beta, sizeof(double) ) );
    CUDA_CALL( cudaMemset( m_d_beta, 0, sizeof(double)) );

    CUDA_CALL( cudaMalloc( (void**)&m_d_eta, sizeof(double) ) );
    CUDA_CALL( cudaMemset( m_d_eta, 0, sizeof(double)) );

    CUDA_CALL( cudaMalloc( (void**)&m_d_mutex, sizeof(int) ) );

    CUDA_CALL( cudaMalloc( (void**)&m_d_lambda_tr, sizeof(double) ) );
    CUDA_CALL( cudaMalloc( (void**)&m_d_lambda_l, sizeof(double) ) );
    CUDA_CALL( cudaMalloc( (void**)&m_d_lambda_u, sizeof(double) ) );
    CUDA_CALL( cudaMalloc( (void**)&m_d_chi_tr, sizeof(double) * m_numElements) );
    CUDA_CALL( cudaMalloc( (void**)&m_d_rho_tr, sizeof(double) ) );
    CUDA_CALL( cudaMalloc( (void**)&m_d_p_w, sizeof(double) ) );

    CUDA_CALL( cudaMemset( m_d_lambda_l, 0, sizeof(double) ) );
    CUDA_CALL( cudaMemset( m_d_lambda_tr, 0, sizeof(double) ) );
    CUDA_CALL( cudaMemset( m_d_lambda_u, 0, sizeof(double) ) );
    CUDA_CALL( cudaMemset( m_d_chi_tr, 0, sizeof(double) * m_numElements) );
    CUDA_CALL( cudaMemset( m_d_rho_tr, 0, sizeof(double) ) );
    CUDA_CALL( cudaMemset( m_d_p_w, 0, sizeof(double) ) );

    return true;
}

bool TDO::innerloop(double* &d_u, double* &d_chi)
{
    m_d_u = d_u;
    m_d_chi = d_chi;
    
    // calculating the driving force of each element
    // df[] = ( 1 / 2*omega ) * ( p * pow(chi[], p - 1 ) ) * sum( u^T * A_local * u )
    // df[] = u^T * A_local * u
    for ( int i = 0 ; i < m_numElements ; i++ )
        calcDrivingForce ( &m_d_df[i], &m_d_chi[i], m_p, m_d_uTAu, m_d_u, m_d_node_index[i], m_d_A_local, m_num_rows, m_gridDim, m_blockDim );

    // DEBUG:
        // cout << "m_d_df" << endl;
        // cudaDeviceSynchronize();
        // printVector_GPU<<<1,m_numElements>>> ( m_d_df, m_numElements );
        // cudaDeviceSynchronize();
        


    // d_temp = u^T * A * u
    vectorEquals_GPU<<<m_gridDim,m_blockDim>>>(m_d_uTAu, m_d_df, m_numElements);

    // NOTE: reduction issue if numElements > blocksize
    calcP_w<<<m_gridDim,m_blockDim>>>(m_d_p_w, m_d_df, m_d_uTAu, m_d_chi, m_p, m_local_volume, m_numElements);

    // print_GPU<<<1,1>>>( m_d_p_w );
    // cudaDeviceSynchronize();


    // calculate eta and beta
    calcEtaBeta<<<1,2>>>( m_d_eta, m_d_beta, m_etastar, m_betastar, m_d_p_w );
    cudaDeviceSynchronize();

    // cout << m_etastar << endl;
    // cout << m_betastar << endl;

    // print_GPU<<<1,1>>>( m_d_beta );
    // cudaDeviceSynchronize();
    // print_GPU<<<1,1>>>( m_d_eta );
    // cudaDeviceSynchronize();


    // NOTE:
    //// for loop
     for ( int j = 0 ; j < m_n ; j++ )
    {

        // df[] = ( 1 / 2*element_volume ) * p * pow(chi_element, (p-1) ) * temp[]
        // temp[] = u[]^T * A * u[]
        UpdateDrivingForce<<<m_gridDim,m_blockDim>>>( m_d_df, m_d_uTAu, m_p, m_d_chi, m_local_volume, m_numElements );

        // printVector_GPU<<<1,m_numElements>>> ( m_d_df, m_numElements );
        // cudaDeviceSynchronize();


        // TODO: laplacian_GPU in these kernels only work on sym. matrices
        // esp for north elements, N*N --> Nx * Ny
        // bisection algo: 
        
        setToZero<<<1,1>>>(m_d_lambda_tr, 1);
        calcLambdaLower<<< m_gridDim, m_blockDim >>> (m_d_df, m_d_lambda_l, m_d_mutex, m_d_beta, m_d_chi, m_d_eta, m_N[0], m_numElements);
        calcLambdaUpper<<< m_gridDim, m_blockDim >>> (m_d_df, m_d_lambda_u, m_d_mutex, m_d_beta, m_d_chi, m_d_eta, m_N[0], m_numElements);
        
        // cudaDeviceSynchronize();
        // cout << "eta, beta" << endl;
        // cudaDeviceSynchronize();
        // print_GPU <<< 1 , 1 >>> ( m_d_eta );
        // cudaDeviceSynchronize();
        // print_GPU <<< 1 , 1 >>> ( m_d_beta );
        // // printVector_GPU<<<1,m_numElements>>> ( m_d_df, m_numElements );

        for ( int i = 1 ; i < 30 ; i++ )
        {
            // cout << "iteration " << i << endl;
            // cudaDeviceSynchronize();
            // cout << "lambda_tr" << endl;
            // print_GPU <<< 1 , 1 >>> ( m_d_lambda_tr );
            // cudaDeviceSynchronize();
            // cout << "lambda_l" << endl;
            // print_GPU <<< 1 , 1 >>> ( m_d_lambda_l );
            // cudaDeviceSynchronize();
            // cout << "lambda_u" << endl;
            // print_GPU <<< 1 , 1 >>> ( m_d_lambda_u );
            // cudaDeviceSynchronize();


            calcChiTrial<<<m_gridDim,m_blockDim>>> ( m_d_chi, m_d_df, m_d_lambda_tr, m_del_t, m_d_eta, m_d_beta, m_d_chi_tr, m_N[0], m_numElements);

            // printVector_GPU<<<1,4>>>( m_d_chi_tr, 4);
            // cudaDeviceSynchronize();


            setToZero<<<1,1>>>(m_d_rho_tr, 1);
            sumOfVector_GPU <<< m_gridDim, m_blockDim >>> (m_d_rho_tr, m_d_chi_tr, m_numElements);
            calcRhoTrial<<<1,1>>>(m_d_rho_tr, m_local_volume, m_numElements);

            // // printVector_GPU<<<m_gridDim,m_blockDim>>>( m_d_chi_tr, m_numElements);
            // print_GPU<<<1,1>>>( m_d_rho_tr );
            // cudaDeviceSynchronize();
            // cout << "\n";


            calcLambdaTrial<<<1,1>>>( m_d_rho_tr, m_rho, m_d_lambda_l, m_d_lambda_u, m_d_lambda_tr);
              

        }


        // chi(j) = chi(j+1)
        vectorEquals_GPU<<<m_gridDim,m_blockDim>>>( m_d_chi, m_d_chi_tr, m_numElements );
       


    }

    

    
    return true;

}