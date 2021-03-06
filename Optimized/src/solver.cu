/*
    solver.cu

    Developed for the master thesis project: GPU-accelerated Thermodynamic Topology Optimization
    Author: Wan Arif bin Wan Abhar
    Institution: Ruhr Universitaet Bochum
*/

#include <iostream>
#include "../include/solver.h"
#include "../include/cudakernels.h"

using namespace std;

Solver::Solver( vector<double*> d_value, vector<size_t*> d_index, vector<size_t> max_row_size, 
                vector<double*> d_p_value, vector<size_t*> d_p_index, vector<size_t> p_max_row_size, 
                size_t numLevels, vector<size_t> num_rows, double damp)
: m_d_value(d_value), m_d_index(d_index), m_max_row_size(max_row_size), m_d_p_value(d_p_value), m_d_p_index(d_p_index), m_p_max_row_size(p_max_row_size), m_numLevels(numLevels), m_num_rows(num_rows), m_damp(damp) 
{
    bm_switch = 0;
    
}

void Solver::set_verbose(bool verbose, bool bs_verbose) { m_verbose = verbose; m_bs_verbose = bs_verbose; }

// DEBUG:
void Solver::set_steps(size_t step, size_t bs_step)
{
    m_step = step;
    m_bs_step = bs_step;
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

void Solver::setBM(bool x) { bm_switch = x; }
float Solver::getCounter(){ return m_counter_it; }



// destructor performs device memory deallocation
Solver::~Solver()
{
    
    CUDA_CALL( cudaFree(m_d_foo) );
    CUDA_CALL( cudaFree(m_d_bs_foo) );
    CUDA_CALL( cudaFree(m_d_r) );
    CUDA_CALL( cudaFree(m_d_c) );
    CUDA_CALL( cudaFree(m_d_res0) );
    CUDA_CALL( cudaFree(m_d_lastRes) );
    CUDA_CALL( cudaFree(m_d_res) );
    CUDA_CALL( cudaFree(m_d_minRes) );
    CUDA_CALL( cudaFree(m_d_minRed) );
    CUDA_CALL( cudaFree(m_d_step) );
    CUDA_CALL( cudaFree(m_d_bs_step) );
    CUDA_CALL( cudaFree(m_d_z) );
    CUDA_CALL( cudaFree(m_d_rho) );
    CUDA_CALL( cudaFree(m_d_rho_old) );
    CUDA_CALL( cudaFree(m_d_p) );
    CUDA_CALL( cudaFree(m_d_alpha) );
    CUDA_CALL( cudaFree(m_d_alpha_temp) );

       
    for ( int i = 0 ; i < m_numLevels ; i++ )
    {
        CUDA_CALL( cudaFree(m_d_gmg_r[i]) );
        CUDA_CALL( cudaFree(m_d_gmg_c[i]) );
        CUDA_CALL( cudaFree(m_d_rtmp[i]) );
        CUDA_CALL( cudaFree(m_d_ctmp[i]) );
    }
    
    CUDA_CALL( cudaFree(m_d_bs_r) );
    CUDA_CALL( cudaFree(m_d_bs_z) );
    CUDA_CALL( cudaFree(m_d_bs_p) );
    CUDA_CALL( cudaFree(m_d_bs_res) );
    CUDA_CALL( cudaFree(m_d_bs_res0) );
    CUDA_CALL( cudaFree(m_d_bs_lastRes) );
    CUDA_CALL( cudaFree(m_d_bs_rho) );
    CUDA_CALL( cudaFree(m_d_bs_rho_old) );
    CUDA_CALL( cudaFree(m_d_bs_alpha) );
    CUDA_CALL( cudaFree(m_d_bs_alpha_temp) );
        
    
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

        m_topLev = m_numLevels - 1;

        // convergence checks
        CUDA_CALL( cudaMalloc((void**)&m_d_foo, sizeof(bool)) );
        CUDA_CALL( cudaMemcpy(m_d_foo, &m_foo, sizeof(bool), cudaMemcpyHostToDevice) );
        CUDA_CALL( cudaMalloc((void**)&m_d_bs_foo, sizeof(bool)) );
        CUDA_CALL( cudaMemcpy(m_d_bs_foo, &m_bs_foo, sizeof(bool), cudaMemcpyHostToDevice) );
        

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
        CUDA_CALL( cudaMalloc((void**)&m_d_minRes, sizeof(double)) );
        CUDA_CALL( cudaMemcpy(m_d_minRes, &m_minRes, sizeof(double), cudaMemcpyHostToDevice) );
        
        // minimum required reduction for convergence
        // d_m_minRed;
        CUDA_CALL( cudaMalloc((void**)&m_d_minRed, sizeof(double)) );
        CUDA_CALL( cudaMemcpy(m_d_minRed, &m_minRed, sizeof(double), cudaMemcpyHostToDevice) );
        
        // steps
        CUDA_CALL( cudaMalloc((void**)&m_d_step, sizeof(size_t)) );
        CUDA_CALL( cudaMemset(m_d_step, 0, sizeof(size_t)) );
        CUDA_CALL( cudaMalloc((void**)&m_d_bs_step, sizeof(size_t)) );
        CUDA_CALL( cudaMemset(m_d_bs_step, 0, sizeof(size_t)) );
        
        // PCG solver
        CUDA_CALL( cudaMalloc((void**)&m_d_z, sizeof(double) * m_num_rows[m_topLev]) );
        CUDA_CALL( cudaMemset(m_d_z, 0, sizeof(double) * m_num_rows[m_topLev]) );
        CUDA_CALL( cudaMalloc((void**)&m_d_rho, sizeof(double)) );
        CUDA_CALL( cudaMemset(m_d_rho, 0, sizeof(double)) );
        CUDA_CALL( cudaMalloc((void**)&m_d_rho_old, sizeof(double)) );
        CUDA_CALL( cudaMemset(m_d_rho_old, 0, sizeof(double)) );
        CUDA_CALL( cudaMalloc((void**)&m_d_p, sizeof(double) * m_num_rows[m_topLev]) );
        CUDA_CALL( cudaMemset(m_d_p, 0, sizeof(double) * m_num_rows[m_topLev]) );
        CUDA_CALL( cudaMalloc((void**)&m_d_alpha, sizeof(double)) );
        CUDA_CALL( cudaMemset(m_d_alpha, 0, sizeof(double)) );
        CUDA_CALL( cudaMalloc((void**)&m_d_alpha_temp, sizeof(double)) );
        CUDA_CALL( cudaMemset(m_d_alpha_temp, 0, sizeof(double)) );
        


        /// GMG precond
        // residuum and correction vectors on each level
        m_d_gmg_r.resize(m_numLevels);
        m_d_gmg_c.resize(m_numLevels);
        
        // temporary residuum vectors for GMG
        m_d_rtmp.resize(m_numLevels);
        
        // temporary correction vectors for GMG
        m_d_ctmp.resize(m_numLevels);
        
        for ( int i = 0 ; i < m_numLevels ; i++ )
        {
            CUDA_CALL( cudaMalloc((void**)&m_d_gmg_r[i], sizeof(double) * m_num_rows[i] ) );
            CUDA_CALL( cudaMalloc((void**)&m_d_gmg_c[i], sizeof(double) * m_num_rows[i] ) );
            CUDA_CALL( cudaMemset(m_d_gmg_r[i], 0, sizeof(double) * m_num_rows[i] ) );
            CUDA_CALL( cudaMemset(m_d_gmg_c[i], 0, sizeof(double) * m_num_rows[i] ) );
            
            CUDA_CALL( cudaMalloc((void**)&m_d_rtmp[i], sizeof(double) * m_num_rows[i] ) );
            CUDA_CALL( cudaMemset(m_d_rtmp[i], 0, sizeof(double) * m_num_rows[i] ) );
            
            CUDA_CALL( cudaMalloc((void**)&m_d_ctmp[i], sizeof(double) * m_num_rows[i] ) );
            CUDA_CALL( cudaMemset(m_d_ctmp[i], 0, sizeof(double) * m_num_rows[i] ) );
        }
        
        
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
        CUDA_CALL( cudaMalloc((void**)&m_d_bs_rho, sizeof(double) ) );
        CUDA_CALL( cudaMemset(m_d_bs_rho, 0, sizeof(double) ) );
        CUDA_CALL( cudaMalloc((void**)&m_d_bs_rho_old, sizeof(double) ) );
        CUDA_CALL( cudaMemset(m_d_bs_rho_old, 0, sizeof(double) ) );
        CUDA_CALL( cudaMalloc((void**)&m_d_bs_alpha, sizeof(double) ) );
        CUDA_CALL( cudaMemset(m_d_bs_alpha, 0, sizeof(double) ) );
        CUDA_CALL( cudaMalloc((void**)&m_d_bs_alpha_temp, sizeof(double) ) );
        CUDA_CALL( cudaMemset(m_d_bs_alpha_temp, 0, sizeof(double) ) );
        

    return true;
}

bool Solver::reinit()
{
        
        setToZero<<<m_gridDim[m_topLev], m_blockDim[m_topLev]>>>( m_d_r, m_num_rows[m_topLev] );
        setToZero<<<m_gridDim[m_topLev], m_blockDim[m_topLev]>>>( m_d_c, m_num_rows[m_topLev] );

        for ( int lev = 0 ; lev < m_numLevels ; lev++ )
        {
            setToZero<<<m_gridDim[lev], m_blockDim[lev]>>>( m_d_gmg_r[lev], m_num_rows[lev] );
            setToZero<<<m_gridDim[lev], m_blockDim[lev]>>>( m_d_gmg_c[lev], m_num_rows[lev] );
            setToZero<<<m_gridDim[lev], m_blockDim[lev]>>>( m_d_rtmp[lev], m_num_rows[lev] );
            setToZero<<<m_gridDim[lev], m_blockDim[lev]>>>( m_d_ctmp[lev], m_num_rows[lev] );
        }

        // scalars
        setToZero<<<1, 1>>>( m_d_res0, 1 );
        setToZero<<<1, 1>>>( m_d_lastRes, 1 );
        setToZero<<<1, 1>>>( m_d_res, 1 );
        setToZero<<<1, 1>>>( m_d_step, 1 );
        setToZero<<<1, 1>>>( m_d_bs_step, 1 );


        // base-solver
        setToZero<<<m_gridDim[0], m_blockDim[0]>>>( m_d_bs_r, m_num_rows[0] );
        setToZero<<<m_gridDim[0], m_blockDim[0]>>>( m_d_bs_z, m_num_rows[0] );
        setToZero<<<m_gridDim[0], m_blockDim[0]>>>( m_d_bs_p, m_num_rows[0] );
        setToZero<<<1, 1>>>( m_d_bs_res, 1 );
        setToZero<<<1, 1>>>( m_d_bs_res0, 1 );
        setToZero<<<1, 1>>>( m_d_bs_lastRes, 1 );
        setToZero<<<1, 1>>>( m_d_bs_rho, 1 );
        setToOne<<<1,1>>>(m_d_bs_rho_old);
        setToZero<<<1, 1>>>( m_d_bs_alpha, 1 );
        setToZero<<<1, 1>>>( m_d_bs_alpha_temp, 1 );


        return true;
}

bool Solver::precond(double* m_d_c, double* m_d_r, ofstream& ofssbm)
{
    
    // reset correction
    // c.resize(d.size()); 
    // c = 0.0;
	setToZero<<<m_gridDim[m_topLev], m_blockDim[m_topLev]>>>(m_d_c, m_num_rows[m_topLev]);

    // Vector<double> rtmp(r);
	vectorEquals_GPU<<<m_gridDim[m_topLev], m_blockDim[m_topLev]>>>(m_d_rtmp[m_topLev], m_d_r, m_num_rows[m_topLev]);
	vectorEquals_GPU<<<m_gridDim[m_topLev], m_blockDim[m_topLev]>>>(m_d_gmg_c[m_topLev], m_d_c, m_num_rows[m_topLev]);
	vectorEquals_GPU<<<m_gridDim[m_topLev], m_blockDim[m_topLev]>>>(m_d_gmg_r[m_topLev], m_d_r, m_num_rows[m_topLev]);

    precond_add_update_GPU(m_d_gmg_c[m_topLev], m_d_rtmp[m_topLev], m_topLev, m_gamma, ofssbm);

    vectorEquals_GPU<<<m_gridDim[m_topLev], m_blockDim[m_topLev]>>>(m_d_c, m_d_gmg_c[m_topLev], m_num_rows[m_topLev]);
	vectorEquals_GPU<<<m_gridDim[m_topLev], m_blockDim[m_topLev]>>>(m_d_r, m_d_gmg_r[m_topLev], m_num_rows[m_topLev]);

    return true;
}

// A*c = r ==> A_coarse*d_bs_u = d_bs_b
bool Solver::base_solve(double* d_bs_u, double* d_bs_b, ofstream& ofssbm)
{
    
    // benchmark output
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds;
    static int base_counter = 0;

    // resetting base solver variables to zero
    setToZero<<<1,1>>>(m_d_bs_step, 1);
    setToTrue<<<1,1>>>(m_d_bs_foo);
    m_bs_foo = true;
    
            
    // m_d_bs_r = d_bs_b - A*d_bs_u
            cudaEventRecord(start);
    ComputeResiduum_GPU<<<m_gridDim[0],m_blockDim[0]>>>(m_num_rows[0], m_max_row_size[0], m_d_value[0], m_d_index[0], d_bs_u, m_d_bs_r, d_bs_b);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            milliseconds = 0;
            cudaEventElapsedTime(&milliseconds, start, stop);
            if ( base_counter == 0 ) ofssbm << "   Base: ComputeResiduum_GPU()\t" << milliseconds << endl;


    // m_d_bs_res = norm(m_d_bs_r)
            cudaEventRecord(start);
    norm_GPU<<<m_gridDim[0], m_blockDim[0]>>>(m_d_bs_res, m_d_bs_r, m_num_rows[0]);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            milliseconds = 0;
            cudaEventElapsedTime(&milliseconds, start, stop);
            if ( base_counter == 0 ) ofssbm << "   Base: norm_GPU()\t\t" << milliseconds << endl;

    equals_GPU<<<1,1>>>(m_d_bs_res0, m_d_bs_res);
    
    if ( m_bs_verbose )
    {
        cout << "CG  : ";
        cudaDeviceSynchronize();
        printInitialResult_GPU<<<1,1>>>(m_d_bs_res0, m_d_minRes, m_d_minRed);
        cudaDeviceSynchronize();
    }
	
 
    // check iteration conditions before the iteration loop
    checkIterationConditionsBS<<<1,1>>>(m_d_bs_foo, m_d_bs_step, m_bs_maxIter, m_d_bs_res, m_d_minRes);
    CUDA_CALL( cudaMemcpy( &m_bs_foo, m_d_bs_foo, sizeof(bool), cudaMemcpyDeviceToHost) 	);
        
    if (!m_bs_foo) return true;

    else
    {
        addStep<<<1,1>>>(m_d_bs_step);

        // iteration loop
        int bs_step = 1;

        while(m_bs_foo || bs_step < m_bs_maxIter)
        {
            // precond, z = r B
                cudaEventRecord(start);
            Jacobi_Precond_GPU<<<m_gridDim[0], m_blockDim[0]>>>(m_d_bs_z, m_d_value[0], m_d_index[0], m_max_row_size[0], m_d_bs_r, m_num_rows[0], m_damp);
                cudaEventRecord(stop);
                cudaEventSynchronize(stop);
                milliseconds = 0;
                cudaEventElapsedTime(&milliseconds, start, stop);
                if ( base_counter == 0 ) ofssbm << "   Base: Jacobi_Precond_GPU()\t" << milliseconds << endl;


            // rho = < z, r >
                cudaEventRecord(start);
            dotProduct(m_d_bs_rho, m_d_bs_r, m_d_bs_z, m_num_rows[0], m_gridDim[0], m_blockDim[0]);
                cudaEventRecord(stop);
                cudaEventSynchronize(stop);
                milliseconds = 0;
                cudaEventElapsedTime(&milliseconds, start, stop);
                if ( base_counter == 0 ) ofssbm << "   Base: dotProduct()\t\t" << milliseconds << endl;


            // calculate p
            // p = z + p * beta;
                cudaEventRecord(start);
            calculateDirectionVector<<<m_gridDim[0],m_blockDim[0]>>>(m_d_bs_step, m_d_bs_p, m_d_bs_z, m_d_bs_rho, m_d_bs_rho_old, m_num_rows[0]);
                cudaEventRecord(stop);
                cudaEventSynchronize(stop);
                milliseconds = 0;
                cudaEventElapsedTime(&milliseconds, start, stop);
                if ( base_counter == 0 ) ofssbm << "   Base: calcDirectionVector()\t" << milliseconds << endl;

            /// z = A*p
                cudaEventRecord(start);
            Apply_GPU<<<m_gridDim[0],m_blockDim[0]>>>( m_num_rows[0], m_max_row_size[0], m_d_value[0], m_d_index[0], m_d_bs_p, m_d_bs_z );
                cudaEventRecord(stop);
                cudaEventSynchronize(stop);
                milliseconds = 0;
                cudaEventElapsedTime(&milliseconds, start, stop);
                if ( base_counter == 0 ) ofssbm << "   Base: Apply_GPU()\t\t" << milliseconds << endl;

            // alpha = rho / (p * z)
                cudaEventRecord(start);
            calculateAlpha(m_d_bs_alpha, m_d_bs_rho, m_d_bs_p, m_d_bs_z, m_d_bs_alpha_temp, m_num_rows[0], m_gridDim[0], m_blockDim[0] );
                cudaEventRecord(stop);
                cudaEventSynchronize(stop);
                milliseconds = 0;
                cudaEventElapsedTime(&milliseconds, start, stop);
                if ( base_counter == 0 ) ofssbm << "   Base: calculateAlpha()\t" << milliseconds << endl;


            // add correction to solution
            // u = u + alpha * p
                cudaEventRecord(start);
            axpy_GPU<<<m_gridDim[0],m_blockDim[0]>>>(d_bs_u, m_d_bs_alpha, m_d_bs_p, m_num_rows[0]);
                cudaEventRecord(stop);
                cudaEventSynchronize(stop);
                milliseconds = 0;
                cudaEventElapsedTime(&milliseconds, start, stop);
                if ( base_counter == 0 ) ofssbm << "   Base: axpy_GPU()\t\t" << milliseconds << endl;

            // update residuum
            // r = r - alpha * z
                cudaEventRecord(start);
            axpy_neg_GPU<<<m_gridDim[0],m_blockDim[0]>>>(m_d_bs_r, m_d_bs_alpha, m_d_bs_z, m_num_rows[0]);
                cudaEventRecord(stop);
                cudaEventSynchronize(stop);
                milliseconds = 0;
                cudaEventElapsedTime(&milliseconds, start, stop);
                if ( base_counter == 0 ) ofssbm << "   Base: axpy_neg_GPU()\t\t" << milliseconds << endl;

            
            // lastRes = res;
            equals_GPU<<<1,1>>>(m_d_bs_lastRes, m_d_bs_res);


            // res = r.norm();
            norm_GPU<<<m_gridDim[0], m_blockDim[0]>>>(m_d_bs_res, m_d_bs_r, m_num_rows[0]);
            
   
    
            // store old rho
            // rho_old = rho;
            vectorEquals_GPU<<<1,1>>>(m_d_bs_rho_old, m_d_bs_rho, 1);


            if ( m_bs_verbose )
            {
                cout << "CG  : ";
                cudaDeviceSynchronize();
                printResult_GPU<<<1,1>>>(m_d_bs_step, m_d_bs_res, m_d_minRes, m_d_bs_lastRes, m_d_bs_res0, m_d_minRed);
                cudaDeviceSynchronize();
            }

            checkIterationConditionsBS<<<1,1>>>(m_d_bs_foo, m_d_bs_step, m_bs_maxIter, m_d_bs_res, m_d_minRes);
    
            CUDA_CALL( cudaMemcpy( &m_bs_foo, m_d_bs_foo, sizeof(bool), cudaMemcpyDeviceToHost) 	);
    
    
            if (!m_bs_foo) break;

            addStep<<<1,1>>>(m_d_bs_step);
    
            bs_step++;
            base_counter++;
        }

        return true;
    
    }
}


bool Solver::precond_add_update_GPU(double* d_c, double* d_r, std::size_t lev, int cycle, ofstream& ofssbm)
{
    // benchmark output
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds;
    static int p_a_u_counter = 0;
    static int p_a_u_counter_ = 0;

    // initialize ctmp[lev] to zero
    setToZero<<< m_gridDim[lev], m_blockDim[lev] >>>( m_d_ctmp[lev], m_num_rows[lev] );			
    

    // if current grid is on base level
	if( lev == 0 )
	{
        base_solve(m_d_ctmp[lev], d_r, ofssbm);   

        // c += ctmp;
		addVector_GPU<<< m_gridDim[lev], m_blockDim[lev] >>>(d_c, m_d_ctmp[lev], m_num_rows[0]);

        // r = r - A[0] * ctmp0
		UpdateResiduum_GPU<<< m_gridDim[lev], m_blockDim[lev] >>>(m_num_rows[lev], m_max_row_size[lev], m_d_value[lev], m_d_index[lev], m_d_ctmp[lev], d_r);

        return true;
    }

    // presmooth
    for ( int i = 0 ; i < m_numPreSmooth ; i++)
    {
        
                cudaEventRecord(start);
        smoother( m_d_ctmp[lev], d_r, lev );
                cudaEventRecord(stop);
                cudaEventSynchronize(stop);
                milliseconds = 0;
                cudaEventElapsedTime(&milliseconds, start, stop);
                if ( p_a_u_counter == 0 && lev == m_topLev && i == 0 ) ofssbm << "Smoother : Jacobi()\t\t" << milliseconds << endl;

        // c += ctmp;
        addVector_GPU<<<m_gridDim[lev], m_blockDim[lev]>>>( d_c, m_d_ctmp[lev], m_num_rows[lev] );
        
        // r -= A[lev] * ctmp;
        UpdateResiduum_GPU<<< m_gridDim[lev], m_blockDim[lev] >>>(m_num_rows[lev], m_max_row_size[lev], m_d_value[lev], m_d_index[lev], m_d_ctmp[lev], d_r);

    }
    
    
    // restrict defect
    setToZero<<<m_gridDim_cols[lev-1],m_blockDim_cols[lev-1]>>>( m_d_gmg_r[lev-1], m_num_rows[lev-1] );
        

    // r_coarse = P^T * r   
            cudaEventRecord(start);
    ApplyTransposed_GPU<<<m_gridDim[lev],m_blockDim[lev]>>>(m_num_rows[lev], m_p_max_row_size[lev-1], m_d_p_value[lev-1], m_d_p_index[lev-1], d_r, m_d_gmg_r[lev-1]);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            milliseconds = 0;
            cudaEventElapsedTime(&milliseconds, start, stop);
            if ( p_a_u_counter == 0 && lev == m_topLev ) ofssbm << "ApplyTransposed_GPU()\t\t" << milliseconds << endl;


    setToZero<<<m_gridDim_cols[lev-1],m_blockDim_cols[lev-1]>>>( m_d_gmg_c[lev-1], m_num_rows[lev-1] );

    
    // F-cycle
    if(cycle == -1) 
	{
        // one F-Cycle ...
        if( !precond_add_update_GPU(m_d_ctmp[lev-1], m_d_rtmp[lev-1], lev-1, -1, ofssbm) )
        {
            std::cout << "gmg failed on level " << lev << ". Aborting." << std::endl;
            return false;
        }

        // ... followed by a V-Cycle
        if( !precond_add_update_GPU(m_d_ctmp[lev-1], m_d_rtmp[lev-1], lev-1, 1, ofssbm) )
        {
            std::cout << "gmg failed on level " << lev << ". Aborting." << std::endl;
            return false;
        }
	}

    // V- and W-cycle
    else
	{
		for (int g = 0; g < cycle; ++g)
		{
			if( !precond_add_update_GPU(m_d_gmg_c[lev-1], m_d_gmg_r[lev-1], lev-1, cycle, ofssbm) )
			{
				std::cout << "gmg failed on level " << lev << ". Aborting." << std::endl;
				return false;
			}
		
		}
    }
    
    /// prolongate coarse grid correction
    // ctmp = P[lev-1] * c_coarse;
            cudaEventRecord(start);
    Apply_GPU<<<m_gridDim[lev],m_blockDim[lev]>>>( m_num_rows[lev], m_p_max_row_size[lev-1], m_d_p_value[lev-1], m_d_p_index[lev-1], m_d_gmg_c[lev-1], m_d_ctmp[lev]);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            milliseconds = 0;
            cudaEventElapsedTime(&milliseconds, start, stop);
            if ( p_a_u_counter_ == 0 && lev == m_topLev )
            {
                ofssbm << "Apply_GPU()\t\t\t" << milliseconds << endl;
                p_a_u_counter_++; 
            }
            
    /// add correction and update defect
	// c += ctmp;
	addVector_GPU<<<m_gridDim[lev],m_blockDim[lev]>>>(d_c, m_d_ctmp[lev], m_num_rows[lev]);

            cudaEventRecord(start); 
    UpdateResiduum_GPU<<<m_gridDim[lev],m_blockDim[lev]>>>( m_num_rows[lev], m_max_row_size[lev] , m_d_value[lev], m_d_index[lev], m_d_ctmp[lev], d_r);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            milliseconds = 0;
            cudaEventElapsedTime(&milliseconds, start, stop);
            if ( p_a_u_counter_ == 1 && lev == m_topLev )
            {
                ofssbm << "UpdateResiduum_GPU()\t\t" << milliseconds << endl;
                p_a_u_counter_++; 
            }
    
    // postsmooth
    for ( int i = 0 ; i < m_numPostSmooth ; i++)
    {
        smoother( m_d_ctmp[lev], d_r, lev );

         // c += ctmp;
        addVector_GPU<<<m_gridDim[lev], m_blockDim[lev]>>>( d_c, m_d_ctmp[lev], m_num_rows[lev] );

        UpdateResiduum_GPU<<< m_gridDim[lev], m_blockDim[lev] >>>(m_num_rows[lev], m_max_row_size[lev], m_d_value[lev], m_d_index[lev], m_d_ctmp[lev], d_r);

    }
    
    if ( lev = m_topLev ) p_a_u_counter++;
    return true;
}

bool Solver::smoother(double* d_c, double* d_r, int lev)
{
    
    Jacobi_Precond_GPU<<<m_gridDim[lev], m_blockDim[lev]>>>(d_c, m_d_value[lev], m_d_index[lev], m_max_row_size[lev], d_r, m_num_rows[lev], m_damp);
    
    return true;
}




bool Solver::solve(double* d_u, double* d_b, vector<double*> d_value, ofstream& ofssbm)
{
    setToOne<<<1,1>>>(m_d_rho_old);

    // benchmark output
    // ofstream ofssbm(filename, ios::out);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds;
    static int bm_counter = 0;
    

    // per iteration
    cudaEvent_t start_it, stop_it;
    cudaEventCreate(&start_it);
    cudaEventCreate(&stop_it);
    float milliseconds_it;
    
    
    // initialization
    setToZero<<<m_gridDim[m_topLev], m_blockDim[m_topLev]>>>( d_u, m_num_rows[m_topLev] );
    setToTrue<<<1,1>>>(m_d_foo);
    m_d_value = d_value;
    m_foo = true;
    
    
    // r = b - A*u
        cudaEventRecord(start);
    ComputeResiduum_GPU<<<m_gridDim[m_topLev], m_blockDim[m_topLev]>>>(m_num_rows[m_topLev], m_max_row_size[m_topLev], m_d_value[m_topLev], m_d_index[m_topLev], d_u, m_d_r, d_b);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        if ( bm_counter == 0 ) ofssbm << "ComputeResiduum_GPU()\t\t" << milliseconds << endl;
    
    
    // d_res0 = norm(m_d_r)
        cudaEventRecord(start);
    norm_GPU<<<m_gridDim[m_topLev], m_blockDim[m_topLev]>>>(m_d_res0, m_d_r, m_num_rows[m_topLev]);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        if ( bm_counter == 0 ) ofssbm << "norm_GPU()\t\t\t" << milliseconds << endl;

    // res = res0;
    equals_GPU<<<1,1>>>(m_d_res, m_d_res0);	

    if ( m_verbose )
    {
        cout << "GMG : ";
        cudaDeviceSynchronize();
        printInitialResult_GPU<<<1,1>>>(m_d_res0, m_d_minRes, m_d_minRed);
        cudaDeviceSynchronize();
    }

    addStep<<<1,1>>>(m_d_step);

    m_sum_it = 0;
    m_counter_it = 0;

    // iteration loop
    while(m_foo)
    {
                
        // GMG-preconditioner
        precond(m_d_z, m_d_r, ofssbm);


        // rho = < z, r >
            cudaEventRecord(start);
        dotProduct(m_d_rho, m_d_r, m_d_z, m_num_rows[m_topLev], m_gridDim[m_topLev], m_blockDim[m_topLev]);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            milliseconds = 0;
            cudaEventElapsedTime(&milliseconds, start, stop);
            if ( bm_counter == 0 ) ofssbm << "dotProduct()\t\t" << milliseconds << endl;

        
        // calculate p
        // p = z + p * (rho / rho_old);
            cudaEventRecord(start);
        calculateDirectionVector<<<m_gridDim[m_topLev],m_blockDim[m_topLev]>>>(m_d_step, m_d_p, m_d_z, m_d_rho, m_d_rho_old, m_num_rows[m_topLev]);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            milliseconds = 0;
            cudaEventElapsedTime(&milliseconds, start, stop);
            if ( bm_counter == 0 ) ofssbm << "calcDirectionVector()\t" << milliseconds << endl;


        /// z = A*p
            cudaEventRecord(start);
        Apply_GPU<<<m_gridDim[m_topLev],m_blockDim[m_topLev]>>>( m_num_rows[m_topLev], m_max_row_size[m_topLev], m_d_value[m_topLev], m_d_index[m_topLev], m_d_p, m_d_z );
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            milliseconds = 0;
            cudaEventElapsedTime(&milliseconds, start, stop);
            if ( bm_counter == 0 ) ofssbm << "Apply_GPU()\t\t" << milliseconds << endl;


        // alpha = rho / (p * z)
            cudaEventRecord(start);
        calculateAlpha(m_d_alpha, m_d_rho, m_d_p, m_d_z, m_d_alpha_temp, m_num_rows[m_topLev], m_gridDim[m_topLev], m_blockDim[m_topLev] );
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            milliseconds = 0;
            cudaEventElapsedTime(&milliseconds, start, stop);
            if ( bm_counter == 0 ) ofssbm << "calculateAlpha()\t" << milliseconds << endl;


        // add correction to solution
        // u = u + alpha * p
            cudaEventRecord(start);
        axpy_GPU<<<m_gridDim[m_topLev],m_blockDim[m_topLev]>>>(d_u, m_d_alpha, m_d_p, m_num_rows[m_topLev]);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            milliseconds = 0;
            cudaEventElapsedTime(&milliseconds, start, stop);
            if ( bm_counter == 0 ) ofssbm << "axpy_GPU()\t\t" << milliseconds << endl;


        // update residuum
        // r = r - alpha * z
            cudaEventRecord(start);
        axpy_neg_GPU<<<m_gridDim[m_topLev],m_blockDim[m_topLev]>>>(m_d_r, m_d_alpha, m_d_z, m_num_rows[m_topLev]);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            milliseconds = 0;
            cudaEventElapsedTime(&milliseconds, start, stop);
            if ( bm_counter == 0 ) ofssbm << "axpy_neg_GPU()\t\t" << milliseconds << endl;

        
        // lastRes = res;
        equals_GPU<<<1,1>>>(m_d_lastRes, m_d_res);


        // res = r.norm();
        norm_GPU<<<m_gridDim[m_topLev], m_blockDim[m_topLev]>>>(m_d_res, m_d_r, m_num_rows[m_topLev]);
   


        // store old rho
        // rho_old = rho;
        vectorEquals_GPU<<<1,1>>>(m_d_rho_old, m_d_rho, 1);

        if ( m_verbose )
        {
        cout << "GMG : ";
        cudaDeviceSynchronize();
        printResult_GPU<<<1,1>>>(m_d_step, m_d_res, m_d_minRes, m_d_lastRes, m_d_res0, m_d_minRed);
        cudaDeviceSynchronize();
        }


        checkIterationConditions<<<1,1>>>(m_d_foo, m_d_step, m_d_res, m_d_res0, m_d_minRes, m_d_minRed, m_maxIter);
        CUDA_CALL( cudaMemcpy( &m_foo, m_d_foo, sizeof(bool), cudaMemcpyDeviceToHost) 	);
        addStep<<<1,1>>>(m_d_step);
        bm_counter++;
        cudaEventRecord(stop_it);
        cudaEventSynchronize(stop_it);
        milliseconds = 0;
        cudaEventElapsedTime(&milliseconds_it, start_it, stop_it);
        m_sum_it += milliseconds_it;
        m_counter_it++;

    }
    
    return true;
    
}


