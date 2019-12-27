#include <iostream>
#include "solver.h"
#include "cudakernels.h"
using namespace std;

Solver::Solver(vector<double*> d_value, vector<size_t*> d_index, vector<size_t> max_row_size, vector<double*> d_p_value, vector<size_t*> d_p_index, vector<size_t> p_max_row_size,double* d_u, double* d_b, size_t numLevels, vector<size_t> num_rows, vector<size_t> num_cols)
: m_d_value(d_value), m_d_index(d_index), m_max_row_size(max_row_size), m_numLevels(numLevels), m_num_rows(num_rows), m_num_cols(num_cols), m_d_p_value(d_p_value), m_d_p_index(d_p_index), m_p_max_row_size(p_max_row_size) 
{
    cout << "constructor" << endl;
    

}

// TODO:
Solver::~Solver()
{
    // CUDA_CALL( cudaFree(m_d_res0) );
    // CUDA_CALL( cudaFree(m_d_res) );
    // CUDA_CALL( cudaFree(m_d_m_minRes) );
    // CUDA_CALL( cudaFree(m_d_m_minRed) );
    // CUDA_CALL( cudaFree(m_d_r) );
    // CUDA_CALL( cudaFree(m_d_c) );
    
    // for ( int i = 0 ; i < m_numLevels ; i++ )
    // {
    //     CUDA_CALL( cudaFree(m_d_rtmp[i]) );
    //     CUDA_CALL( cudaFree(m_d_ctmp[i]) );
    //     CUDA_CALL( cudaFree(m_d_gmg_r[i]) );
    //     CUDA_CALL( cudaFree(m_d_gmg_c[i]) );
    // }
}

void Solver::set_num_prepostsmooth(size_t pre_n, size_t post_n)
{
    m_numPreSmooth = pre_n;
    m_numPostSmooth = post_n;
}


void Solver::set_cycle(const char type)
{
    switch(type){
        case 'V': m_gamma = 1; break;
        case 'W': m_gamma = 2; break;
        case 'F': m_gamma = -1; break;
        
        default: std::cout << "Cycle type '" << type << "' invalid argument" << std::endl;
        throw std::invalid_argument("Cycle type: invalid argument");
    }
}
    
bool Solver::init()
{
        cout << "init" << endl;

        m_topLev = m_numLevels - 1;
        
        // calculate cuda grid and block dimensions of each level

        m_gridDim.resize(m_numLevels);
        m_blockDim.resize(m_numLevels);
        // TODO: for loop
        calculateDimensions(8, m_gridDim[0], m_blockDim[0]);
        calculateDimensions(18, m_gridDim[1], m_blockDim[1]);
    
		    
		m_gridDim_cols.resize(m_numLevels - 1);
        m_blockDim_cols.resize(m_numLevels - 1);
		calculateDimensions(8, m_gridDim_cols[0], m_blockDim_cols[0]);



        // TODO: store each level's num_rows in an array
        // e.g., m_num_rows[lev]


        CUDA_CALL( cudaMalloc((void**)&m_d_r, sizeof(double) * m_num_rows[m_topLev]) );
        CUDA_CALL( cudaMemset(m_d_r, 0, sizeof(double) * m_num_rows[m_topLev]) );
        CUDA_CALL( cudaMalloc((void**)&m_d_c, sizeof(double) * m_num_rows[m_topLev]) );
        CUDA_CALL( cudaMemset(m_d_c, 0, sizeof(double) * m_num_rows[m_topLev]) );


        // previous residuum
        CUDA_CALL( cudaMalloc((void**)&m_d_res0, sizeof(double)) );
        CUDA_CALL( cudaMemset(m_d_res0, 0, sizeof(double)) );
        
        // current residuum
        CUDA_CALL( cudaMalloc((void**)&m_d_res, sizeof(double)) );
        CUDA_CALL( cudaMemset(m_d_res, 0, sizeof(double)) );
    
        // minimum required residuum for convergence
        // d_m_minRes;
        CUDA_CALL( cudaMalloc((void**)&m_d_m_minRes, sizeof(double)) );
        CUDA_CALL( cudaMemset(m_d_m_minRes, 1.000e-99, sizeof(double)) );
        
        // minimum required reduction for convergence
        // d_m_minRed;
        CUDA_CALL( cudaMalloc((void**)&m_d_m_minRed, sizeof(double)) );
        CUDA_CALL( cudaMemset(m_d_m_minRed, 1.000e-10, sizeof(double)) );
        

        /// GMG precond
        // residuum and correction vectors on each level
        m_d_gmg_r.resize(m_numLevels);
        m_d_gmg_c.resize(m_numLevels);
        CUDA_CALL( cudaMalloc((void**)&m_d_gmg_r[0], sizeof(double) * m_num_rows[0] ) );
        CUDA_CALL( cudaMalloc((void**)&m_d_gmg_r[1], sizeof(double) * m_num_rows[1] ) );
        CUDA_CALL( cudaMalloc((void**)&m_d_gmg_c[0], sizeof(double) * m_num_rows[0] ) );
        CUDA_CALL( cudaMalloc((void**)&m_d_gmg_c[1], sizeof(double) * m_num_rows[1] ) );

        CUDA_CALL( cudaMemset(m_d_gmg_r[0], 0, sizeof(double) * m_num_rows[0] ) );
        CUDA_CALL( cudaMemset(m_d_gmg_r[1], 0, sizeof(double) * m_num_rows[1] ) );
        CUDA_CALL( cudaMemset(m_d_gmg_c[0], 0, sizeof(double) * m_num_rows[0] ) );
        CUDA_CALL( cudaMemset(m_d_gmg_c[1], 0, sizeof(double) * m_num_rows[1] ) );


        // temporary residuum vectors for GMG
        m_d_rtmp.resize(m_numLevels);
        CUDA_CALL( cudaMalloc((void**)&m_d_rtmp[0], sizeof(double) * m_num_rows[0] ) );
        CUDA_CALL( cudaMalloc((void**)&m_d_rtmp[1], sizeof(double) * m_num_rows[1] ) );
        CUDA_CALL( cudaMemset(m_d_rtmp[0], 0, sizeof(double) * m_num_rows[0] ) );
        CUDA_CALL( cudaMemset(m_d_rtmp[1], 0, sizeof(double) * m_num_rows[1] ) );
    
        // temporary correction vectors for GMG
        m_d_ctmp.resize(m_numLevels);
        CUDA_CALL( cudaMalloc((void**)&m_d_ctmp[0], sizeof(double) * m_num_rows[0] ) );
        CUDA_CALL( cudaMalloc((void**)&m_d_ctmp[1], sizeof(double) * m_num_rows[1] ) );
        CUDA_CALL( cudaMemset(m_d_ctmp[0], 0, sizeof(double) * m_num_rows[0] ) );
        CUDA_CALL( cudaMemset(m_d_ctmp[1], 0, sizeof(double) * m_num_rows[1] ) );

    return true;
}

bool Solver::precond(double* d_c, double* d_r)
{
    cout << "precond" << endl;
    cudaDeviceSynchronize();

    // reset correction
    // c.resize(d.size()); 
    // c = 0.0;
	setToZero<<<m_gridDim[m_topLev], m_blockDim[m_topLev]>>>(d_c, m_num_rows[m_topLev]);

    // Vector<double> rtmp(r);
	vectorEquals_GPU<<<m_gridDim[m_topLev], m_blockDim[m_topLev]>>>(m_d_rtmp[m_topLev], d_r, m_num_rows[m_topLev]);

    
	// NOTE: the original d_c and d_r from i_s.cu stay here
	// d_gmg_c[topLev] = d_c
	// d_gmg_r[topLev] = d_r
	vectorEquals_GPU<<<m_gridDim[m_topLev], m_blockDim[m_topLev]>>>(m_d_gmg_c[m_topLev], d_c, m_num_rows[m_topLev]);
	cudaDeviceSynchronize();
	vectorEquals_GPU<<<m_gridDim[m_topLev], m_blockDim[m_topLev]>>>(m_d_gmg_r[m_topLev], d_r, m_num_rows[m_topLev]);
	cudaDeviceSynchronize();

    precond_add_update_GPU(m_d_gmg_c[m_topLev], m_d_rtmp[m_topLev], m_topLev, m_gamma);

    return true;
}

bool Solver::precond_add_update_GPU(double* d_c, double* d_r, std::size_t lev, int cycle)
{
    cout << "precond_add_update" << endl;

    // std::cout <<"gmg.cu : setToZero()" << std::endl;
    // Vector<double> ctmp(c.size(), 0.0, c.layouts());
    setToZero<<< m_gridDim[lev], m_blockDim[lev] >>>( m_d_ctmp[lev], 18 );			
    cudaDeviceSynchronize();


    // if on base level
	if( lev == 0 )
	{
        cout << "base level" << endl;

    }

    
    // presmooth
    
    for ( int i = 0 ; i < m_numPreSmooth ; i++)
    {
        smoother( m_d_ctmp[lev], d_r, lev );

         // c += ctmp;
        addVector_GPU<<<m_gridDim[lev], m_blockDim[lev]>>>( d_c, m_d_ctmp[lev], m_num_rows[lev] );

        UpdateResiduum_GPU<<< m_gridDim[lev], m_blockDim[lev] >>>(m_num_rows[lev], m_max_row_size[lev], m_d_value[lev], m_d_index[lev], m_d_ctmp[lev], d_r);

    }

    // restrict defect
    setToZero<<<m_gridDim_cols[lev-1],m_blockDim_cols[lev-1]>>>( m_d_gmg_r[lev-1], m_num_rows[lev-1] );

    /// r_coarse = P^T * r




    return true;
}

bool Solver::smoother(double* d_c, double* d_r, int lev)
{
    
    
    // cout << "smoother" << endl;

    // Jacobi_Precond_GPU<<<m_gridDim[lev], m_blockDim[lev]>>>(d_c, m_d_value, m_d_index, m_max_row_size, d_r, m_num_rows);

   

    return true;
}

bool Solver::solve(double* d_u, double* d_b)
{
    cout << "solve" << endl;

    // r = b - A*u
    ComputeResiduum_GPU<<<m_gridDim[m_topLev], m_blockDim[m_topLev]>>>(m_num_rows[m_topLev], m_max_row_size[m_topLev], m_d_value[m_topLev], m_d_index[m_topLev], d_u, m_d_r, d_b);
    
    // d_res0 = norm(m_d_r)
    norm_GPU(m_d_res0, m_d_r, m_num_rows[m_topLev], m_gridDim[m_topLev], m_blockDim[m_topLev]);
    
    // res = res0;
    equals_GPU<<<1,1>>>(m_d_res, m_d_res0);	


    printInitialResult_GPU<<<1,1>>>(m_d_res0, m_d_m_minRes, m_d_m_minRed);
    cudaDeviceSynchronize();
    

    // foo loop

    precond(m_d_c, m_d_r);


    return true;
}

// cudaDeviceSynchronize();
// print_GPU<<<1,1>>>( d_res0 );