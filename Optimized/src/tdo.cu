/*
    tdo.cu

    Developed for the master thesis project: GPU-accelerated Thermodynamic Topology Optimization
    Author: Wan Arif bin Wan Abhar
    Institution: Ruhr Universitaet Bochum
*/

#include <iostream>
#include <cmath>
#include "../include/assemble.h"
#include "../include/cudakernels.h"
#include "../include/tdo.h"

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


void TDO::setBM(bool x){ bm_switch = x; }
int TDO::getCounter(){ return m_counter; }
float TDO::getSum(){ return m_sum_it; }


TDO::TDO(double* d_u, double* d_chi, double h, size_t dim, double betastar, double etastar, size_t numElements, size_t num_rows, double* d_A_local, vector<size_t*> d_node_index, vector<size_t> N, double rho, size_t numLevels, size_t p, size_t* &d_node_index_)
 : m_d_u(d_u), m_d_chi(d_chi), m_h(h), m_dim(dim), m_numElements(numElements), m_num_rows(num_rows), m_d_A_local(d_A_local), m_d_node_index(d_node_index), m_rho(rho), m_etastar(etastar), m_betastar(betastar), m_numLevels(numLevels), m_p(p), m_d_node_index_(d_node_index_)
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
    m_local_volume = pow(m_h, m_dim); 
    bm_switch = 0;
    
}

// destructor performs device memory deallocation
TDO::~TDO()
{
    CUDA_CALL( cudaFree(m_d_df) );
    CUDA_CALL( cudaFree(m_d_beta) );
    CUDA_CALL( cudaFree(m_d_eta) );
    CUDA_CALL( cudaFree(m_d_mutex) );
    CUDA_CALL( cudaFree(m_d_lambda_tr) );
    CUDA_CALL( cudaFree(m_d_lambda_l) );
    CUDA_CALL( cudaFree(m_d_lambda_u) );
    CUDA_CALL( cudaFree(m_d_chi_tr) );
    CUDA_CALL( cudaFree(m_d_rho_tr) );
    CUDA_CALL( cudaFree(m_d_p_w) );
    CUDA_CALL( cudaFree(m_d_tdo_foo) );
    CUDA_CALL( cudaFree(m_d_sum_g) );
    CUDA_CALL( cudaFree(m_d_sum_df_g) );

}


bool TDO::init()
{

    calculateDimensions(m_numElements, m_gridDim, m_blockDim);
        
    CUDA_CALL( cudaMalloc( (void**)&m_d_df, sizeof(double) * m_numElements ) );
    CUDA_CALL( cudaMemset( m_d_df, 0, sizeof(double) * m_numElements) );

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
    

    CUDA_CALL( cudaMalloc( (void**)&m_d_sum_g, sizeof(double) ) );
    CUDA_CALL( cudaMalloc( (void**)&m_d_sum_df_g, sizeof(double) ) );

    return true;
}

void TDO::set_verbose(bool verbose) { m_verbose = verbose; }
void TDO::print_VTK(bool foo) { m_printVTK = foo; }

bool TDO::innerloop(double* &d_u, double* &d_chi, double* &d_c, double* &d_MOD, ofstream& ofssbm)
{
    // benchmark output
    // ofstream ofssbm(filename, ios::out);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds;
    static int inner_counter = 0;

    // per iteration
    cudaEvent_t start_it, stop_it;
    cudaEventCreate(&start_it);
    cudaEventCreate(&stop_it);
    float milliseconds_it;
    
    m_d_u = d_u;
    m_d_chi = d_chi;
    m_tdo_foo = true;
    setToTrue<<<1,1>>>( m_d_tdo_foo );
    setToZero<<<1,1>>>( m_d_p_w, 1 );
    setToZero<<<1,1>>>( m_d_sum_g, 1 );
    setToZero<<<1,1>>>( m_d_sum_df_g, 1 );
    setToZero<<<1,1>>>( m_d_df, m_numElements );
    
    
    
    //// loop n times
    for ( int j = 0 ; j < m_n ; j++ )
    {

        // calculating the driving force of each element
        // df[] = ( 1 / 2*local_volume ) * ( p * pow(chi[], p - 1 ) ) * ( u^T * A_local * u )
            cudaEventRecord(start);
                calcDrivingForce<<<m_gridDim, m_blockDim>>>(m_d_df, m_d_u, m_d_chi, m_p, m_d_node_index_, m_d_A_local, m_num_rows, m_dim, m_local_volume, m_numElements);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            milliseconds = 0;
            cudaEventElapsedTime(&milliseconds, start, stop);
            if ( inner_counter == 0 ) ofssbm << "calcDrivingForce()\t\t" << milliseconds << endl;

        // calculating average weighted driving force, p_w
            cudaEventRecord(start);
        calcP_w_(m_d_p_w, m_d_sum_g, m_d_sum_df_g, m_d_df, m_d_chi, m_numElements, m_local_volume);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            milliseconds = 0;
            cudaEventElapsedTime(&milliseconds, start, stop);
            if ( inner_counter == 0 ) ofssbm << "calcP_w()\t\t\t" << milliseconds << endl;

        // calculating eta and beta
            cudaEventRecord(start);
        calcEtaBeta<<<1,2>>>( m_d_eta, m_d_beta, m_etastar, m_betastar, m_d_p_w );
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            milliseconds = 0;
            cudaEventElapsedTime(&milliseconds, start, stop);
            if ( inner_counter == 0 ) ofssbm << "calcEtaBeta()\t\t\t" << milliseconds << endl;

        // bisection algo: 
        setToZero<<<1,1>>>(m_d_lambda_tr, 1);
            cudaEventRecord(start);

        // computing lambda lower
        calcLambdaLower<<< m_gridDim, m_blockDim >>> (m_d_df, m_d_lambda_l, m_d_mutex, m_d_beta, m_d_chi, m_d_eta, m_Nx, m_Ny, m_Nz, m_numElements, m_h);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            milliseconds = 0;
            cudaEventElapsedTime(&milliseconds, start, stop);
            if ( inner_counter == 0 ) ofssbm << "calcLambdaLower()\t\t" << milliseconds << endl;
        
        // computing lambda upper
            cudaEventRecord(start);
        calcLambdaUpper<<< m_gridDim, m_blockDim >>> (m_d_df, m_d_lambda_u, m_d_mutex, m_d_beta, m_d_chi, m_d_eta, m_Nx, m_Ny, m_Nz, m_numElements, m_h);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            milliseconds = 0;
            cudaEventElapsedTime(&milliseconds, start, stop);
            if ( inner_counter == 0 ) ofssbm << "calcLambdaLower()\t\t" << milliseconds << endl;        
        
        // remaining operations on lambda lower and upper
        minus_GPU<<<1,1>>>( m_d_lambda_l, m_d_eta);
        add_GPU<<<1,1>>>( m_d_lambda_u, m_d_eta);
        

        m_sum_it = 0;
        m_counter = 0;
        
        while(m_tdo_foo)
        {
            
            cudaEventRecord(start_it);
            // computing chi_trial
                cudaEventRecord(start);
            calcChiTrial<<<m_gridDim,m_blockDim>>> ( m_d_chi, m_d_df, m_d_lambda_tr, m_del_t, m_d_eta, m_d_beta, m_d_chi_tr, m_Nx, m_Ny, m_Nz, m_numElements, m_h);
                cudaEventRecord(stop);
                cudaEventSynchronize(stop);
                milliseconds = 0;
                cudaEventElapsedTime(&milliseconds, start, stop);
                if ( inner_counter == 0 ) ofssbm << "calcChiTrial()\t\t\t" << milliseconds << endl;    

            // computing rho_trial
            setToZero<<<1,1>>>(m_d_rho_tr, 1);
                cudaEventRecord(start);
            sumOfVector_GPU <<< m_gridDim, m_blockDim >>> (m_d_rho_tr, m_d_chi_tr, m_numElements);
            calcRhoTrial<<<1,1>>>(m_d_rho_tr, m_local_volume, m_numElements);
                cudaEventRecord(stop);
                cudaEventSynchronize(stop);
                milliseconds = 0;
                cudaEventElapsedTime(&milliseconds, start, stop);
                if ( inner_counter == 0 ) ofssbm << "calcRhoTrial()\t\t\t" << milliseconds << endl;    

                cudaEventRecord(start);
            calcLambdaTrial<<<1,1>>>( m_d_rho_tr, m_rho, m_d_lambda_l, m_d_lambda_u, m_d_lambda_tr);
                cudaEventRecord(stop);
                cudaEventSynchronize(stop);
                milliseconds = 0;
                cudaEventElapsedTime(&milliseconds, start, stop);
                if ( inner_counter == 0 ) ofssbm << "calcLambdaTrial()\t\t" << milliseconds << endl;    

            checkTDOConvergence<<<1,1>>> ( m_d_tdo_foo, m_rho, m_d_rho_tr);
            CUDA_CALL( cudaMemcpy( &m_tdo_foo, m_d_tdo_foo, sizeof(bool), cudaMemcpyDeviceToHost) 	);

            
            cudaEventRecord(stop_it);
            cudaEventSynchronize(stop_it);
            milliseconds = 0;
            cudaEventElapsedTime(&milliseconds_it, start_it, stop_it);
            m_sum_it += milliseconds_it;
            inner_counter++;
            m_counter++;
        }
        
        // computing compliance, c = 0.5 * sum( u^T * K * u )
        setToZero<<<1,1>>> ( d_c, 1 );
        calcCompliance<<< m_gridDim, m_blockDim >>> (d_c, m_d_u, d_chi, m_d_node_index_, m_d_A_local, m_local_volume, m_num_rows, m_dim, m_numElements);
        
        
        // computing MOD
        setToZero<<<1,1>>> ( d_MOD, 1 );
        calcMOD<<< m_gridDim, m_blockDim >>> (d_MOD, d_chi, m_local_volume, m_numElements);
        
        
        if ( bm_switch == 0) ofssbm << "Total time of bisection algo \t" << m_sum_it << endl;
        if ( bm_switch == 0) ofssbm << "Number of steps \t\t" << inner_counter << endl;
        if ( bm_switch == 0) ofssbm << "Average time per bisection step " << m_sum_it/inner_counter << endl;

        // chi(j) = chi(j+1)
        vectorEquals_GPU<<<m_gridDim,m_blockDim>>>( m_d_chi, m_d_chi_tr, m_numElements );
        
    }
    
    return true;

}

