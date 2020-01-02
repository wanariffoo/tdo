#include <iostream>
#include "solver.h"
#include "cudakernels.h"
using namespace std;

Solver::Solver(vector<double*> d_value, vector<size_t*> d_index, vector<double*> d_p_value, vector<size_t*> d_p_index, size_t numLevels, vector<size_t> num_rows, vector<size_t> max_row_size, vector<size_t> p_max_row_size, double damp)
: m_d_value(d_value), m_d_index(d_index), m_d_p_value(d_p_value), m_d_p_index(d_p_index), m_numLevels(numLevels), m_num_rows(num_rows), m_max_row_size(max_row_size), m_p_max_row_size(p_max_row_size), m_damp(damp) 
{
    cout << "solver : constructor" << endl;

    // m_numLevels = numLevels;
    // , m_num_rows(num_rows), m_max_row_size(max_row_size), m_p_max_row_size(p_max_row_size) 

    // printVector_GPU <<< 1 , num_rows[1] * max_row_size[1] >>> ( d_value[1], num_rows[1] * max_row_size[1] );


    // m_d_value.resize(1);
    // m_d_value = d_value;

    // printVector_GPU<<<1,15>>>( m_d_value[0], 15);
    // printVector_GPU<<<1,15>>>( m_d_value[1], 15);

}

// TODO: could try as a destructor
// void Solver::deallocate()
Solver::~Solver()
{
    cout << "solver : deallocate" << endl;
    CUDA_CALL( cudaFree(m_d_res0) );
    CUDA_CALL( cudaFree(m_d_res) );
    CUDA_CALL( cudaFree(m_d_lastRes) );
    CUDA_CALL( cudaFree(m_d_m_minRes) );
    CUDA_CALL( cudaFree(m_d_m_minRed) );
    CUDA_CALL( cudaFree(m_d_r) );
    CUDA_CALL( cudaFree(m_d_c) );
    CUDA_CALL( cudaFree(m_d_step) );
    CUDA_CALL( cudaFree(m_d_bs_step) );
    
    // base solver
    CUDA_CALL( cudaFree(m_d_bs_r) );
    CUDA_CALL( cudaFree(m_d_bs_z) );
    CUDA_CALL( cudaFree(m_d_bs_res) );
    CUDA_CALL( cudaFree(m_d_bs_lastRes) );
    CUDA_CALL( cudaFree(m_d_bs_res0) );
    CUDA_CALL( cudaFree(m_d_bs_m_minRes) );
    CUDA_CALL( cudaFree(m_d_bs_m_minRed) );
    CUDA_CALL( cudaFree(m_d_bs_rho_old) );
    CUDA_CALL( cudaFree(m_d_bs_p) );
    CUDA_CALL( cudaFree(m_d_bs_alpha) );
    CUDA_CALL( cudaFree(m_d_bs_alpha_temp) );
    
}

void Solver::set_num_prepostsmooth(size_t pre_n, size_t post_n)
{
    m_numPreSmooth = pre_n;
    m_numPostSmooth = post_n;
}


void Solver::set_convergence_params( size_t maxIter, double minRes, double minRed )
{
	m_maxIter = maxIter;
	m_minRes = minRes;
	m_minRed = minRed;
}

void Solver::set_bs_convergence_params( size_t maxIter, double minRes, double minRed )
{
	m_bs_maxIter = maxIter;
	m_bs_minRes = minRes;
	m_bs_minRed = minRed;
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
        cout << "solver : init" << endl;

        m_topLev = m_numLevels - 1;
        
        // calculate cuda grid and block dimensions of each level

        m_gridDim.resize(m_numLevels);
        m_blockDim.resize(m_numLevels);

        for ( int i = 0 ; i < m_numLevels ; i++ )
            calculateDimensions(m_num_rows[i], m_gridDim[i], m_blockDim[i]);
		    
		m_gridDim_cols.resize(m_numLevels - 1);
        m_blockDim_cols.resize(m_numLevels - 1);

        for ( int i = 0 ; i < m_numLevels - 1; i++ )
		    calculateDimensions(m_num_rows[i], m_gridDim_cols[i], m_blockDim_cols[i]);

        

        CUDA_CALL( cudaMalloc((void**)&m_d_r, sizeof(double) * m_num_rows[m_topLev]) );
        CUDA_CALL( cudaMemset(m_d_r, 0, sizeof(double) * m_num_rows[m_topLev]) );
        CUDA_CALL( cudaMalloc((void**)&m_d_c, sizeof(double) * m_num_rows[m_topLev]) );
        CUDA_CALL( cudaMemset(m_d_c, 0, sizeof(double) * m_num_rows[m_topLev]) );

        // TODO: perhaps you could use a temp variable here, no need to malloc?
        // temp residuum
        CUDA_CALL( cudaMalloc((void**)&m_d_res0, sizeof(double)) );
        CUDA_CALL( cudaMemset(m_d_res0, 0, sizeof(double)) );

        // last residuum
        CUDA_CALL( cudaMalloc((void**)&m_d_lastRes, sizeof(double)) );
        CUDA_CALL( cudaMemset(m_d_lastRes, 0, sizeof(double)) );
        
        // current residuum
        CUDA_CALL( cudaMalloc((void**)&m_d_res, sizeof(double)) );
        CUDA_CALL( cudaMemset(m_d_res, 0, sizeof(double)) );
    
        // minimum required residuum for convergence
        // d_m_minRes;
        CUDA_CALL( cudaMalloc((void**)&m_d_m_minRes, sizeof(double)) );
        CUDA_CALL( cudaMemset(m_d_m_minRes, m_minRes, sizeof(double)) );
        
        // minimum required reduction for convergence
        // d_m_minRed;
        CUDA_CALL( cudaMalloc((void**)&m_d_m_minRed, sizeof(double)) );
        CUDA_CALL( cudaMemset(m_d_m_minRed, m_minRed, sizeof(double)) );
        
        // steps
        CUDA_CALL( cudaMalloc((void**)&m_d_step, sizeof(size_t)) );
        CUDA_CALL( cudaMemset(m_d_step, 0, sizeof(size_t)) );
        CUDA_CALL( cudaMalloc((void**)&m_d_bs_step, sizeof(size_t)) );
        CUDA_CALL( cudaMemset(m_d_bs_step, 0, sizeof(size_t)) );

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



        // base-solver

        CUDA_CALL( cudaMalloc((void**)&m_d_bs_r, sizeof(double) * m_num_rows[0] ) );
        CUDA_CALL( cudaMemset(m_d_bs_r, 0, sizeof(double) * m_num_rows[0] ) );
        CUDA_CALL( cudaMalloc((void**)&m_d_bs_z, sizeof(double) * m_num_rows[0] ) );
        CUDA_CALL( cudaMemset(m_d_bs_z, 0, sizeof(double) * m_num_rows[0] ) );
        CUDA_CALL( cudaMalloc((void**)&m_d_bs_p, sizeof(double) * m_num_rows[0] ) );
        CUDA_CALL( cudaMemset(m_d_bs_p, 0, sizeof(double) * m_num_rows[0] ) );

        CUDA_CALL( cudaMalloc((void**)&m_d_bs_res, sizeof(double) ) );
        CUDA_CALL( cudaMemset(m_d_bs_res, 0, sizeof(double) ) );
        CUDA_CALL( cudaMalloc((void**)&m_d_bs_res0, sizeof(double) ) );
        CUDA_CALL( cudaMemset(m_d_bs_res0, 0, sizeof(double) ) );
        CUDA_CALL( cudaMalloc((void**)&m_d_bs_lastRes, sizeof(double) ) );
        CUDA_CALL( cudaMemset(m_d_bs_lastRes, 0, sizeof(double) ) );
        CUDA_CALL( cudaMalloc((void**)&m_d_bs_m_minRes, sizeof(double) ) );
        CUDA_CALL( cudaMemset(m_d_bs_m_minRes, 0, sizeof(double) ) );
        CUDA_CALL( cudaMalloc((void**)&m_d_bs_m_minRed, sizeof(double) ) );
        CUDA_CALL( cudaMemset(m_d_bs_m_minRed, 0, sizeof(double) ) );
        CUDA_CALL( cudaMalloc((void**)&m_d_bs_rho, sizeof(double) ) );
        CUDA_CALL( cudaMemset(m_d_bs_rho, 0, sizeof(double) ) );
        CUDA_CALL( cudaMalloc((void**)&m_d_bs_rho_old, sizeof(double) ) );
        CUDA_CALL( cudaMemset(m_d_bs_rho_old, 0, sizeof(double) ) );
        CUDA_CALL( cudaMalloc((void**)&m_d_bs_alpha, sizeof(double) ) );
        CUDA_CALL( cudaMemset(m_d_bs_alpha, 0, sizeof(double) ) );
        CUDA_CALL( cudaMalloc((void**)&m_d_bs_alpha_temp, sizeof(double) ) );
        CUDA_CALL( cudaMemset(m_d_bs_alpha_temp, 0, sizeof(double) ) );
        CUDA_CALL( cudaMalloc((void**)&m_d_bs_alpha_temp, sizeof(double) ) );
        CUDA_CALL( cudaMemset(m_d_bs_alpha_temp, 0, sizeof(double) ) );
        CUDA_CALL( cudaMalloc((void**)&m_d_bs_m_minRed, sizeof(double) ) );
        CUDA_CALL( cudaMemset(m_d_bs_m_minRed, 1e-10, sizeof(double) ) );
        CUDA_CALL( cudaMalloc((void**)&m_d_bs_m_minRes, sizeof(double) ) );
        CUDA_CALL( cudaMemset(m_d_bs_m_minRes, 1e-99, sizeof(double) ) );
       

    return true;
}

bool Solver::precond(double* m_d_c, double* m_d_r)
{
    cout << "precond" << endl;
    cudaDeviceSynchronize();

    // reset correction
    // c.resize(d.size()); 
    // c = 0.0;
	setToZero<<<m_gridDim[m_topLev], m_blockDim[m_topLev]>>>(m_d_c, m_num_rows[m_topLev]);

    

    // Vector<double> rtmp(r);
	vectorEquals_GPU<<<m_gridDim[m_topLev], m_blockDim[m_topLev]>>>(m_d_rtmp[m_topLev], m_d_r, m_num_rows[m_topLev]);

    
	// NOTE: the original d_c and d_r from i_s.cu stay here
	// d_gmg_c[topLev] = d_c
	// d_gmg_r[topLev] = d_r
	vectorEquals_GPU<<<m_gridDim[m_topLev], m_blockDim[m_topLev]>>>(m_d_gmg_c[m_topLev], m_d_c, m_num_rows[m_topLev]);
	cudaDeviceSynchronize();
	vectorEquals_GPU<<<m_gridDim[m_topLev], m_blockDim[m_topLev]>>>(m_d_gmg_r[m_topLev], m_d_r, m_num_rows[m_topLev]);
	cudaDeviceSynchronize();

    precond_add_update_GPU(m_d_gmg_c[m_topLev], m_d_rtmp[m_topLev], m_topLev, m_gamma);

    vectorEquals_GPU<<<m_gridDim[m_topLev], m_blockDim[m_topLev]>>>(m_d_c, m_d_gmg_c[m_topLev], m_num_rows[m_topLev]);
	// cudaDeviceSynchronize();
	vectorEquals_GPU<<<m_gridDim[m_topLev], m_blockDim[m_topLev]>>>(m_d_r, m_d_gmg_r[m_topLev], m_num_rows[m_topLev]);

    return true;
}

bool Solver::base_solve(double* d_bs_u, double* d_bs_b)
{
    cout << "CG" << endl;

    ComputeResiduum_GPU<<<m_gridDim[0],m_blockDim[0]>>>(m_num_rows[0], m_max_row_size[0], m_d_value[0], m_d_index[0], d_bs_u, m_d_bs_r, d_bs_b);

    norm_GPU(m_d_bs_res, m_d_bs_r, m_num_rows[0], m_gridDim[0], m_blockDim[0]);


    equals_GPU<<<1,1>>>(m_d_bs_res0, m_d_bs_res);
    
    // cudaDeviceSynchronize();
    // cout << "## CG  ##################################################################" << endl;
    // cout << "  Iter     Residuum       Required       Rate        Reduction     Required" << endl;
    // cudaDeviceSynchronize();
	

	
    // printInitialResult_GPU<<<1,1>>>(m_d_bs_res0, m_d_bs_m_minRes, m_d_bs_m_minRed);
    // cudaDeviceSynchronize();
    
    addStep<<<1,1>>>(m_d_bs_step);

    // TODO: add this before foo loop
    // checkIterationConditions<<<1,1>>>(d_cg_foo, d_cg_step, d_cg_res, d_cg_res0, d_cg_m_minRes, d_cg_m_minRed, d_cg_m_maxIter);

    // foo loop
    for ( int i = 0 ; i < 30 ; i++ )
    {
    // TODO: check
    // smoother( m_d_bs_z, m_d_bs_r, 0);
    
    vectorEquals_GPU<<<m_gridDim[0],m_blockDim[0]>>>(m_d_bs_z, m_d_bs_r, m_num_rows[0]);

    // rho = < z, r >
    dotProduct(m_d_bs_rho, m_d_bs_r, m_d_bs_z, m_num_rows[0], m_gridDim[0], m_blockDim[0]);

    calculateDirectionVector<<<m_gridDim[0],m_blockDim[0]>>>(m_d_bs_step, m_d_bs_p, m_d_bs_z, m_d_bs_rho, m_d_bs_rho_old, m_num_rows[0]);

    /// z = A*p
    Apply_GPU<<<m_gridDim[0],m_blockDim[0]>>>( m_num_rows[0], m_max_row_size[0], m_d_value[0], m_d_index[0], m_d_bs_p, m_d_bs_z );

    // alpha = rho / (p * z)
    calculateAlpha(m_d_bs_alpha, m_d_bs_rho, m_d_bs_p, m_d_bs_z, m_d_bs_alpha_temp, m_num_rows[0], m_gridDim[0], m_blockDim[0] );

    // add correction to solution
    // x = x + alpha * p
    axpy_GPU<<<m_gridDim[0],m_blockDim[0]>>>(d_bs_u, m_d_bs_alpha, m_d_bs_p, m_num_rows[0]);

    // update residuum
    // r = r - alpha * z
    axpy_neg_GPU<<<m_gridDim[0],m_blockDim[0]>>>(m_d_bs_r, m_d_bs_alpha, m_d_bs_z, m_num_rows[0]);
    
    // compute residuum
    // lastRes = res;
    equals_GPU<<<1,1>>>(m_d_bs_lastRes, m_d_bs_res);
    
    // res = r.norm();
    norm_GPU(m_d_bs_res, m_d_bs_r, m_num_rows[0], m_gridDim[0], m_blockDim[0]);

    // store old rho
    // rho_old = rho;
    vectorEquals_GPU<<<m_gridDim[0],m_blockDim[0]>>>(m_d_bs_rho_old, m_d_bs_rho, m_num_rows[0]);

    // cudaDeviceSynchronize();
    // printResult_GPU<<<1,1>>>(m_d_bs_step, m_d_bs_res, m_d_bs_m_minRes, m_d_bs_lastRes, m_d_bs_res0, m_d_bs_m_minRed);
    // cudaDeviceSynchronize();
    
    addStep<<<1,1>>>(m_d_bs_step);
    }

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
        base_solve(m_d_ctmp[lev], d_r);

        // c += ctmp;
		addVector_GPU<<< m_gridDim[lev], m_blockDim[lev] >>>(d_c, m_d_ctmp[lev], m_num_rows[0]);

        // r -= A[0] * c;
		UpdateResiduum_GPU<<< m_gridDim[lev], m_blockDim[lev] >>>(m_num_rows[lev], m_max_row_size[lev], m_d_value[lev], m_d_index[lev], m_d_ctmp[lev], d_r);
		// cudaDeviceSynchronize();


        return true;
    }

    
    // presmooth
    
    for ( int i = 0 ; i < m_numPreSmooth ; i++)
    {
        smoother( m_d_ctmp[lev], d_r, lev );
        
        // c += ctmp;
        addVector_GPU<<<m_gridDim[lev], m_blockDim[lev]>>>( d_c, m_d_ctmp[lev], m_num_rows[lev] );
        
        
        // r -= A[lev] * ctmp;
        UpdateResiduum_GPU<<< m_gridDim[lev], m_blockDim[lev] >>>(m_num_rows[lev], m_max_row_size[lev], m_d_value[lev], m_d_index[lev], m_d_ctmp[lev], d_r);
       
        // DEBUG: debugged until here in Octave
        // cudaDeviceSynchronize();
        // printVector_GPU<<< 1, 18 >>>( d_r,18 );
        // cudaDeviceSynchronize();
    }

    
    // restrict defect
    setToZero<<<m_gridDim_cols[lev-1],m_blockDim_cols[lev-1]>>>( m_d_gmg_r[lev-1], m_num_rows[lev-1] );

    /// r_coarse = P^T * r
    ApplyTransposed_GPU<<<m_gridDim[lev-1],m_blockDim[lev-1]>>>(m_num_rows[lev-1], m_max_row_size[lev-1], m_d_p_value[lev-1], m_d_p_index[lev-1], d_r, m_d_gmg_r[lev-1]);

    setToZero<<<m_gridDim_cols[lev-1],m_blockDim_cols[lev-1]>>>( m_d_gmg_c[lev-1], m_num_rows[lev-1] );

    
    if(cycle == -1) // F-cycle
	{

		// cout << "F cycle" << endl; // DEBUG:
		// cudaDeviceSynchronize();
			// one F-Cycle ...
		    if( !precond_add_update_GPU(m_d_ctmp[lev-1], m_d_rtmp[lev-1], lev-1, -1) )  // TODO: check ctmp or gmg_c?
			{
		        std::cout << "gmg failed on level " << lev << ". Aborting." << std::endl;
		        return false;
		    }

		    // ... followed by a V-Cycle
		   	if( !precond_add_update_GPU(m_d_ctmp[lev-1], m_d_rtmp[lev-1], lev-1, 1) )
			{
		        std::cout << "gmg failed on level " << lev << ". Aborting." << std::endl;
		        return false;
			}
	}

    else
	{

		// V- and W-cycle
		for (int g = 0; g < cycle; ++g)
		{

			if( !precond_add_update_GPU(m_d_gmg_c[lev-1], m_d_gmg_r[lev-1], lev-1, cycle) )
			{
				std::cout << "gmg failed on level " << lev << ". Aborting." << std::endl;
				return false;
			}
		
		}
    }
    
    /// prolongate coarse grid correction
	// ctmp = P[lev-1] * c_coarse;
    Apply_GPU<<<m_gridDim[lev-1],m_blockDim[lev-1]>>>( m_num_rows[lev-1], m_p_max_row_size[lev-1], m_d_p_value[lev-1], m_d_p_index[lev-1], m_d_gmg_c[lev-1], m_d_ctmp[lev]);


    /// add correction and update defect
	// c += ctmp;
	addVector_GPU<<<m_gridDim[lev],m_blockDim[lev]>>>(d_c, m_d_ctmp[lev], m_num_rows[lev]);
    
    UpdateResiduum_GPU<<<m_gridDim[lev],m_blockDim[lev]>>>( m_num_rows[lev], m_max_row_size[lev] , m_d_value[lev], m_d_index[lev], m_d_ctmp[lev], d_r);
    
    
    // postsmooth
    for ( int i = 0 ; i < m_numPostSmooth ; i++)
    {
        smoother( m_d_ctmp[lev], d_r, lev );

         // c += ctmp;
        addVector_GPU<<<m_gridDim[lev], m_blockDim[lev]>>>( d_c, m_d_ctmp[lev], m_num_rows[lev] );

        UpdateResiduum_GPU<<< m_gridDim[lev], m_blockDim[lev] >>>(m_num_rows[lev], m_max_row_size[lev], m_d_value[lev], m_d_index[lev], m_d_ctmp[lev], d_r);

    }


    return true;
}

bool Solver::smoother(double* d_c, double* d_r, int lev)
{
    
    cout << "smoother" << endl;

    Jacobi_Precond_GPU<<<m_gridDim[lev], m_blockDim[lev]>>>(d_c, m_d_value[lev], m_d_index[lev], m_max_row_size[lev], d_r, m_num_rows[lev], m_damp);

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
    
    int step = 20;

    // foo loop
    for (int i = 0 ; i < step ; i++ )
    {

    precond(m_d_c, m_d_r);

    // add correction to solution
    // u += c;
    addVector_GPU<<<m_gridDim[m_topLev], m_blockDim[m_topLev]>>>( d_u, m_d_c, m_num_rows[m_topLev] );

    // update residuum r = r - A*c
    UpdateResiduum_GPU<<<m_gridDim[m_topLev], m_blockDim[m_topLev]>>>( m_num_rows[m_topLev], m_max_row_size[m_topLev], m_d_value[m_topLev], m_d_index[m_topLev], m_d_c, m_d_r );

    // remember last residuum norm
    // lastRes = res;
    equals_GPU<<<1,1>>>(m_d_lastRes, m_d_res);

    // compute new residuum norm
    // res = r.norm();
    // norm_GPU<<<gridDim,blockDim>>>(d_res, d_r, A.num_rows());
    // TODO:
    norm_GPU(m_d_res, m_d_r, m_num_rows[m_topLev], m_gridDim[m_topLev], m_blockDim[m_topLev]);

    printResult_GPU<<<1,1>>>(m_d_step, m_d_res, m_d_m_minRes, m_d_lastRes, m_d_res0, m_d_m_minRed);

    }



    return true;
}

// cudaDeviceSynchronize();
// print_GPU<<<1,1>>>( d_res0 );
// printVector_GPU<<<1,18>>>( d_c, 18 );

   
// cudaDeviceSynchronize();
// printVector_GPU<<< 1, 1 >>>( m_d_res0,1 );
// cudaDeviceSynchronize();