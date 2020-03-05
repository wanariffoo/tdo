
#include <iostream>
#include "assemble.h"
#include <cmath>
#include "cudakernels.h"
#include "tdo.h"

#include <fstream>
#include <stdexcept>
#include <sstream>
#include <string>

using namespace std;

void WriteVectorToVTK_df(vector<double> &df, vector<double> &u, const std::string& filename, size_t dim, vector<size_t> numNodesPerDim, double h, size_t numElements, size_t numNodes)
{
	
	std::ofstream ofs(filename, std::ios::out);
	if (ofs.bad())
	{
		std::ostringstream oss;
		oss << "File '" << filename << "' could not be opened for writing.";
		throw std::runtime_error(oss.str());
	}

	ofs << "# vtk DataFile Version 2.0" << std::endl;
	ofs << "Thermodynamics Topology Optimzation" << std::endl;
	ofs << "ASCII" << std::endl;
	ofs << endl;
	ofs << "DATASET STRUCTURED_GRID" << std::endl;

	if ( dim == 2 )
		numNodesPerDim.push_back(1);

	// specify number of nodes in each dimension
	ofs << "DIMENSIONS";
	for (std::size_t i = 0; i < 3; ++i)
		ofs << " " << numNodesPerDim[i];
	// for (std::size_t i = dim; i < 3; ++i)
	// 		ofs << " " << 1;
	ofs << std::endl;

	// specify the coordinates of all points
	ofs << "POINTS ";
	ofs << numNodes << " float" << endl;

    if ( dim == 2)
    {
        for (std::size_t z = 0; z < numNodesPerDim[2]; ++z)
        {
            for (std::size_t y = 0; y < numNodesPerDim[1]; ++y)
            {
                for (std::size_t x = 0; x < numNodesPerDim[0]; ++x)
                    ofs << " " << h*x << " " << h*z << " " << h*y << endl;
            }
        }
    }

    else
    {
        for (std::size_t z = 0; z < numNodesPerDim[2]; ++z)
        {
            for (std::size_t y = 0; y < numNodesPerDim[1]; ++y)
            {
                for (std::size_t x = 0; x < numNodesPerDim[0]; ++x)
                    ofs << " " << h*x << " " << h*y << " " << h*z << endl;
            }
        }
    }

	ofs << endl;

	// specifying the design variable in each element
	ofs << "CELL_DATA " << numElements << endl;
	ofs << "SCALARS df double" << endl;
	ofs << "LOOKUP_TABLE default" << endl;

	for (int i = 0 ; i < numElements ; i++)
		ofs << " " << df[i] << endl;

	ofs << endl;

	// specifying the displacements for all dimensions in each point
	ofs << "POINT_DATA " << numNodes << std::endl;
	ofs << "VECTORS displacements double" << std::endl;

	
	for ( int i = 0 ; i < numNodes ; i++ )
	{
		if ( dim == 2 )
		{
			for ( int j = 0 ; j < 2 ; j++ )
				ofs << " " << u[dim*i + j];

			// setting displacement in z-dimension to zero
			ofs << " 0";
		}

		else
		{
			for ( int j = 0 ; j < dim ; j++ )
				ofs << " " << u[dim*i + j];
		}

		ofs << endl;
	}


	
}


void WriteVectorToVTK_laplacian(vector<double> &laplacian, vector<double> &u, const std::string& filename, size_t dim, vector<size_t> numNodesPerDim, double h, size_t numElements, size_t numNodes)
{
	
	std::ofstream ofs(filename, std::ios::out);
	if (ofs.bad())
	{
		std::ostringstream oss;
		oss << "File '" << filename << "' could not be opened for writing.";
		throw std::runtime_error(oss.str());
	}

	ofs << "# vtk DataFile Version 2.0" << std::endl;
	ofs << "Thermodynamics Topology Optimzation" << std::endl;
	ofs << "ASCII" << std::endl;
	ofs << endl;
	ofs << "DATASET STRUCTURED_GRID" << std::endl;

	if ( dim == 2 )
		numNodesPerDim.push_back(1);

	// specify number of nodes in each dimension
	ofs << "DIMENSIONS";
	for (std::size_t i = 0; i < 3; ++i)
		ofs << " " << numNodesPerDim[i];
	// for (std::size_t i = dim; i < 3; ++i)
	// 		ofs << " " << 1;
	ofs << std::endl;

	// specify the coordinates of all points
	ofs << "POINTS ";
	ofs << numNodes << " float" << endl;

	for (std::size_t z = 0; z < numNodesPerDim[2]; ++z)
	{
		for (std::size_t y = 0; y < numNodesPerDim[1]; ++y)
		{
			for (std::size_t x = 0; x < numNodesPerDim[0]; ++x)
				ofs << " " << h*x << " " << h*z << " " << h*y << endl;
		}
	}

	ofs << endl;

    // specifying the laplacian in each element
    ofs << "CELL_DATA " << numElements << endl;
	ofs << "SCALARS lp double" << endl;
	ofs << "LOOKUP_TABLE default" << endl;

	for (int i = 0 ; i < numElements ; i++)
		ofs << " " << laplacian[i] << endl;

	ofs << endl;

	// specifying the displacements for all dimensions in each point
	ofs << "POINT_DATA " << numNodes << std::endl;
	ofs << "VECTORS displacements double" << std::endl;

	
	for ( int i = 0 ; i < numNodes ; i++ )
	{
		if ( dim == 2 )
		{
			for ( int j = 0 ; j < 2 ; j++ )
				ofs << " " << u[dim*i + j];

			// setting displacement in z-dimension to zero
			ofs << " 0";
		}

		else
		{
			for ( int j = 0 ; j < dim ; j++ )
				ofs << " " << u[dim*i + j];
		}

		ofs << endl;
	}


	
}





TDO::TDO(double* d_u, double* d_chi, double h, size_t dim, double betastar, double etastar, size_t numElements, size_t num_rows, double* d_A_local, vector<size_t*> d_node_index, vector<size_t> N, double rho, size_t numLevels, size_t p)
 : m_d_u(d_u), m_d_chi(d_chi), m_h(h), m_dim(dim), m_numElements(numElements), m_num_rows(num_rows), m_d_A_local(d_A_local), m_d_node_index(d_node_index), m_rho(rho), m_etastar(etastar), m_betastar(betastar), m_numLevels(numLevels), m_p(p)
{
    // inner loop frequency, n
    m_n = (6 / m_etastar) * ( m_betastar / (m_h*m_h) );
    m_del_t = 1.0 / m_n;


    m_Nx = N[0];
    m_Ny = N[1];

    if (N.size() == 3)
        m_Nz = N[2];
    else
        m_Nz = 0;
    
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
    
    CUDA_CALL( cudaMalloc( (void**)&m_d_temp_s, sizeof(double) ));
    CUDA_CALL( cudaMemset( m_d_temp_s, 0, sizeof(double) ) );

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

    CUDA_CALL( cudaMalloc( (void**)&m_d_tdo_foo, sizeof(bool) ) );
    CUDA_CALL( cudaMemcpy( m_d_tdo_foo, &m_tdo_foo, sizeof(bool), cudaMemcpyHostToDevice) );
    


    // DEBUG: temporary
    CUDA_CALL( cudaMalloc( (void**)&d_laplacian, sizeof(double) * m_numElements) );
    CUDA_CALL( cudaMemset( d_laplacian, 0, sizeof(double) * m_numElements) );

    CUDA_CALL( cudaMalloc( (void**)&m_d_sum_g, sizeof(double) ) );
    CUDA_CALL( cudaMalloc( (void**)&m_d_sum_df_g, sizeof(double) ) );





    return true;
}

void TDO::set_verbose(bool verbose) { m_verbose = verbose; }

bool TDO::innerloop(double* &d_u, double* &d_chi)
{
    
    
    m_d_u = d_u;
    m_d_chi = d_chi;
    m_tdo_foo = true;
    setToTrue<<<1,1>>>( m_d_tdo_foo );
    laplacian.resize(m_numElements);
    setToZero<<<m_gridDim,m_blockDim>>>( d_laplacian, m_numElements );
    setToZero<<<1,1>>>( m_d_sum_g, 1 );
    setToZero<<<1,1>>>( m_d_sum_df_g, 1 );
    
    

    // calculating the driving force of each element
    // df[] = ( 1 / 2*local_volume ) * ( p * pow(chi[], p - 1 ) ) * ( u^T * A_local * u )
    calcDrivingForce ( m_d_df, m_d_chi, m_p, m_d_uTAu, m_d_u, m_d_node_index, m_d_A_local, m_num_rows, m_gridDim, m_blockDim, m_dim, m_numElements, m_local_volume );

   

    // DEBUG:
    checkLaplacian<<<m_gridDim,m_blockDim>>>( d_laplacian, m_d_chi, m_Nx, m_Ny, m_Nz, m_numElements, m_h );

    // if(m_verbose)
    // printVector_GPU<<<m_gridDim,m_blockDim>>>( m_d_df, m_numElements);
    // printVector_GPU<<<m_gridDim,m_blockDim>>>( m_d_chi, m_numElements);
    // printVector_GPU<<<m_gridDim,m_blockDim>>>( d_laplacian, m_numElements);
    
    // print_GPU<<<1,1>>>( &m_d_df[168] );
    cudaDeviceSynchronize();



    // getting vtk for driving force
    vector<size_t> numNodesPerDim(3);
    numNodesPerDim[0] = m_Nx + 1;
    numNodesPerDim[1] = m_Ny + 1;
    numNodesPerDim[2] = m_Nz + 1;
    size_t numNodes = numNodesPerDim[0] * numNodesPerDim[1] * numNodesPerDim[2];
    
    vector<double> df_(m_numElements, 0);
    vector<double> u(numNodes * m_dim, 0);
    string fileformat(".vtk");
    stringstream ss_; 
    ss_ << "vtk/df";
    ss_ << m_file_index;
    ss_ << fileformat;
    stringstream ss__; 
    ss__ << "vtk/laplacian";
    ss__ << m_file_index;
    ss__ << fileformat;
    CUDA_CALL( cudaMemcpy(&u[0], d_u, sizeof(double) * numNodes * m_dim, cudaMemcpyDeviceToHost) );
    CUDA_CALL( cudaMemcpy( &df_[0], m_d_df, sizeof(double) * m_numElements, cudaMemcpyDeviceToHost) );
    CUDA_CALL( cudaMemcpy( &laplacian[0], d_laplacian, sizeof(double) * m_numElements, cudaMemcpyDeviceToHost) );

    
    WriteVectorToVTK_df(df_, u, ss_.str(), m_dim, numNodesPerDim, m_h, m_numElements, numNodes );
    // WriteVectorToVTK_laplacian(laplacian, u, ss__.str(), m_dim, numNodesPerDim, m_h, m_numElements, numNodes );
    ++m_file_index;
    
    calcP_w(m_d_p_w, m_d_sum_g, m_d_sum_df_g, m_d_df, m_d_chi, m_d_temp, m_d_temp_s, m_numElements, m_local_volume);


    // calculate eta and beta
    calcEtaBeta<<<1,2>>>( m_d_eta, m_d_beta, m_etastar, m_betastar, m_d_p_w );
    cudaDeviceSynchronize();


    
    // // cout << "aps" << endl;
    // // cout << m_etastar << endl;
    // // cout << m_betastar << endl;

    // print_GPU<<<1,1>>>( m_d_p_w );
    // // cout << "sum_g" << endl;
    // cudaDeviceSynchronize();
    // // print_GPU<<<1,1>>>( m_d_temp );
    // cudaDeviceSynchronize();
    // print_GPU<<<1,1>>>( m_d_beta );
    // // cudaDeviceSynchronize();
    // // print_GPU<<<1,1>>>( m_d_tdo_foo );
    // // cudaDeviceSynchronize();

    


    //// for loop
    for ( int j = 0 ; j < m_n ; j++ )
    {

        // bisection algo: 
        setToZero<<<1,1>>>(m_d_lambda_tr, 1);
        calcLambdaLower<<< m_gridDim, m_blockDim >>> (m_d_df, m_d_lambda_l, m_d_mutex, m_d_beta, m_d_chi, m_d_eta, m_Nx, m_Ny, m_Nz, m_numElements, m_h);
        calcLambdaUpper<<< m_gridDim, m_blockDim >>> (m_d_df, m_d_lambda_u, m_d_mutex, m_d_beta, m_d_chi, m_d_eta, m_Nx, m_Ny, m_Nz, m_numElements, m_h);
        

        minus_GPU<<<1,1>>>( m_d_lambda_l, m_d_eta);
        add_GPU<<<1,1>>>( m_d_lambda_u, m_d_eta);

            // print_GPU<<<1,1>>>( m_d_lambda_l );
            // cudaDeviceSynchronize();
            // print_GPU<<<1,1>>>( m_d_lambda_u );
            // cudaDeviceSynchronize();

        int counter = 0;
        while(m_tdo_foo)
        {
            
            calcChiTrial<<<m_gridDim,m_blockDim>>> ( m_d_chi, m_d_df, m_d_lambda_tr, m_del_t, m_d_eta, m_d_beta, m_d_chi_tr, m_Nx, m_Ny, m_Nz, m_numElements, m_h);

            // print_GPU<<<1,1>>>( &m_d_chi_tr[0] );
            // print_GPU<<<1,1>>>( &m_d_chi_tr[0] );
            cudaDeviceSynchronize();
            // print_GPU<<<1,1>>>( &m_d_chi_tr[168] );
            // print_GPU<<<1,1>>>( &m_d_chi_tr[168] );
            cudaDeviceSynchronize();

            setToZero<<<1,1>>>(m_d_rho_tr, 1);
            sumOfVector_GPU <<< m_gridDim, m_blockDim >>> (m_d_rho_tr, m_d_chi_tr, m_numElements);
            calcRhoTrial<<<1,1>>>(m_d_rho_tr, m_local_volume, m_numElements);

            if ( counter == 0 )
            {
            // printVector_GPU<<<m_gridDim,m_blockDim>>>( m_d_chi_tr, m_numElements);
            // print_GPU<<<1,1>>>( m_d_rho_tr );

            }
            cudaDeviceSynchronize();
            // cout << "\n";

            calcLambdaTrial<<<1,1>>>( m_d_rho_tr, m_rho, m_d_lambda_l, m_d_lambda_u, m_d_lambda_tr);

            checkTDOConvergence<<<1,1>>> ( m_d_tdo_foo, m_rho, m_d_rho_tr);
            CUDA_CALL( cudaMemcpy( &m_tdo_foo, m_d_tdo_foo, sizeof(bool), cudaMemcpyDeviceToHost) 	);

            // cout << "counter " << ++counter << endl;
        }

        // printVector_GPU<<<1,10>>>(m_d_chi_tr, 10);

        // chi(j) = chi(j+1)
        vectorEquals_GPU<<<m_gridDim,m_blockDim>>>( m_d_chi, m_d_chi_tr, m_numElements );
       
    }


        // if(m_verbose)
        // printVector_GPU<<<1,m_numElements>>>( m_d_chi, m_numElements);   
        // bar<<<1,1>>>( m_d_chi );    

    
    return true;

}

    // cudaDeviceSynchronize();
    // cout << "aps" << endl;
    // print_GPU<<<1,1>>>( m_d_p_w );