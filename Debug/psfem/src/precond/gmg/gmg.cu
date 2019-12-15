/*
 * gmg.cpp
 *
 * author: a.vogel@rub.de
 *
 */

#include "gmg.h"
#include <cstddef>
#include <cassert>
#include <iostream>

using namespace std;


template<std::size_t dim>
GMG<dim>::GMG
(
	StructuredMultiGrid<dim>& multiGrid,
	IAssemble<dim>& disc,
	IProlongation<dim>& prol,
	LinearIterator& smoother,
	LinearSolver& baseSolver
)
: m_multiGrid(multiGrid), m_pDisc(&disc),
  m_pProl(&prol), m_pSmoother(&smoother), m_pBaseSolver(&baseSolver)
{
	// default number of smoothing steps
	set_num_smooth(1);
	set_cycle(1);
    set_base_level(0);
    set_rap(false);
}

template<std::size_t dim>
void GMG<dim>::set_rap(bool bRAP){
	m_bRAP = bRAP;
}

template<std::size_t dim>
void GMG<dim>::set_base_level(std::size_t lvl){

	if(lvl >= m_multiGrid.num_levels()){
		std::cout << "BaseLvl = " << lvl << " requested, but only " << m_multiGrid.num_levels() << " available." << std::endl;
        throw std::invalid_argument("BaseLvl: invalid argument");
	}

	m_baseLvl = lvl;
}


template<std::size_t dim>
void GMG<dim>::set_cycle(const char type){
    switch(type){
        case 'V': m_gamma = 1; break;
        case 'W': m_gamma = 2; break;
        case 'F': m_gamma = -1; break;
            
        default: std::cout << "Cycle type '" << type << "' invalid argument" << std::endl;
            throw std::invalid_argument("Cycle type: invalid argument");
    }
}


template<std::size_t dim>
void GMG<dim>::set_cycle(int gamma){
    if(gamma < 1){
        std::cout << "Gamma = " << gamma << " is invalid argument" << std::endl;
        throw std::invalid_argument("Cycle type: invalid argument");
    }
    m_gamma = gamma;
}


template<std::size_t dim>
bool GMG<dim>::init(const ELLMatrix<double>& mat)
{
	// // DEBUG:
	// std::cout << "gmg.cu : init()" << std::endl;
	// std::cout << "gmg.cu : prolong matrix " << std::endl;
	
	// assemble prolongation
	m_vProlongMat.resize(m_multiGrid.num_levels() - 1);
	for(std::size_t lev = 0; lev < m_vProlongMat.size(); ++lev){
		m_pProl->assemble(m_vProlongMat[lev], m_multiGrid.grid(lev+1), m_multiGrid.grid(lev));
	}
	
	// std::cout << "gmg.cu : coarse matrix " << std::endl;
	// create coarse grid matrices
	m_vStiffMat.resize(m_multiGrid.num_levels());
	if(m_bRAP){
		
		// copy finest matrix
		m_vStiffMat[m_multiGrid.num_levels() - 1] = mat;
		
		// use P^T A P
		for(std::size_t lev = m_vStiffMat.size()-1; lev != 0; --lev){
			MultiplyPTAP(m_vStiffMat[lev-1], m_vStiffMat[lev], m_vProlongMat[lev-1]);
			m_vStiffMat[lev-1].set_storage_type(PST_ADDITIVE);
			m_vStiffMat[lev-1].set_layouts(m_multiGrid.grid(lev-1).layouts());
		}
		
	} else {
		
		// assemble matrices
		m_vStiffMat.resize(m_multiGrid.num_levels());
		Vector<double> dummyX;
		for(std::size_t lev = 0; lev < m_vStiffMat.size(); ++lev)
		m_pDisc->assemble(m_vStiffMat[lev], dummyX, m_multiGrid.grid(lev));
	}
	
	
	// std::cout << "gmg.cu : init basesolver()" << std::endl;
	// init base solver
	m_pBaseSolver->init(m_vStiffMat[m_baseLvl]);

	// init smoother
	m_vSmoother.resize(m_multiGrid.num_levels());
	for(std::size_t lev = 0; lev < m_vSmoother.size(); ++lev)
	{
		m_vSmoother[lev].reset(m_pSmoother->clone());
		m_vSmoother[lev]->init(m_vStiffMat[lev]);
	}

	// CUDA
	// Allocating and copying the relevant data to device
		
		d_gmg_c.resize(m_multiGrid.num_levels());
		d_gmg_r.resize(m_multiGrid.num_levels());
		d_ctmp.resize(m_multiGrid.num_levels());
		d_rtmp.resize(m_multiGrid.num_levels());
		d_value.resize(m_multiGrid.num_levels());
		d_index.resize(m_multiGrid.num_levels());
		d_p_value.resize(m_multiGrid.num_levels() - 1);
		d_p_index.resize(m_multiGrid.num_levels() - 1);

	// TODO: maybe this is unnecessary?
		std::vector<double> ctmp(1, 0);
		std::vector<double> rtmp(1, 0);

	// DEBUG: dummy variable for debugging
		
		CUDA_CALL(cudaMalloc((void **)&d_aps, sizeof(double)));
		CUDA_CALL(cudaMemcpy(d_aps, &appie, sizeof(double), cudaMemcpyHostToDevice));

	// Allocating and copying the required value and index vectors of the stiffness matrices of all levels
	for (int i = 0; i < m_multiGrid.num_levels(); ++i)
	{
		// value vector
		CUDA_CALL(cudaMalloc((void **)&d_value[i], sizeof(double) * m_vStiffMat[i].num_rows() * m_vStiffMat[i].max_row_size()));
		cudaDeviceSynchronize();
		CUDA_CALL(cudaMemcpy(d_value[i], m_vStiffMat[i].getValueAddress(), sizeof(double) * m_vStiffMat[i].num_rows() * m_vStiffMat[i].max_row_size(), cudaMemcpyHostToDevice));
		cudaDeviceSynchronize();
		
		// index vector
		CUDA_CALL(cudaMalloc((void **)&d_index[i], m_vStiffMat[i].num_rows() * m_vStiffMat[i].max_row_size() * sizeof(size_t)));
		cudaDeviceSynchronize();
		CUDA_CALL(cudaMemcpy(d_index[i], m_vStiffMat[i].getIndexAddress(), sizeof(size_t) * m_vStiffMat[i].num_rows() * m_vStiffMat[i].max_row_size(), cudaMemcpyHostToDevice));
		cudaDeviceSynchronize();

		// correction vector
		ctmp.resize(m_vStiffMat[i].num_rows(), 0);
		CUDA_CALL(cudaMalloc((void **)&d_gmg_c[i], sizeof(double) * m_vStiffMat[i].num_rows()));
		CUDA_CALL(cudaMemcpy(d_gmg_c[i], &ctmp[0], sizeof(double) * m_vStiffMat[i].num_rows(), cudaMemcpyHostToDevice));

		// residuum vector
		rtmp.resize(m_vStiffMat[i].num_rows(), 0);
		CUDA_CALL(cudaMalloc((void **)&d_gmg_r[i], sizeof(double) * m_vStiffMat[i].num_rows()));
		CUDA_CALL(cudaMemcpy(d_gmg_r[i], &ctmp[0], sizeof(double) * m_vStiffMat[i].num_rows(), cudaMemcpyHostToDevice));	

		// temporary correction vector
		CUDA_CALL(cudaMalloc((void **)&d_ctmp[i], sizeof(double) * m_vStiffMat[i].num_rows()));
		CUDA_CALL(cudaMemcpy(d_ctmp[i], &ctmp[0], sizeof(double) * m_vStiffMat[i].num_rows(), cudaMemcpyHostToDevice));

		// temporary residuum vector
		CUDA_CALL(cudaMalloc((void **)&d_rtmp[i], sizeof(double) * m_vStiffMat[i].num_rows()));
		CUDA_CALL(cudaMemcpy(d_rtmp[i], &rtmp[0], sizeof(double) * m_vStiffMat[i].num_rows(), cudaMemcpyHostToDevice));

		cudaDeviceSynchronize(); // NOTE: synchronize to prevent "stack smashing" error
	}


	// Allocating and copying the required value and index vectors of the prolongation matrices of all levels
	for ( int i = 0 ; i < m_multiGrid.num_levels() - 1 ; ++i )
	{
		// value vector
		CUDA_CALL( cudaMalloc( (void**)&d_p_value[i], sizeof(double) * m_vProlongMat[i].num_rows() * m_vProlongMat[i].max_row_size() ) 	);
		// cudaDeviceSynchronize(); // TODO: remove?
		CUDA_CALL( cudaMemcpy( d_p_value[i], m_vProlongMat[i].getValueAddress(), sizeof(double) * m_vProlongMat[i].num_rows() * m_vProlongMat[i].max_row_size(), cudaMemcpyHostToDevice ) 		);
		// cudaDeviceSynchronize(); // TODO: remove?
		
		// index vector
		CUDA_CALL( cudaMalloc( (void**)&d_p_index[i], m_vProlongMat[i].num_rows() * m_vProlongMat[i].max_row_size() * sizeof(size_t) ) 	);
		// cudaDeviceSynchronize(); // TODO: remove?
		CUDA_CALL( cudaMemcpy( d_p_index[i], m_vProlongMat[i].getIndexAddress(), sizeof(size_t) * m_vProlongMat[i].num_rows() * m_vProlongMat[i].max_row_size(), cudaMemcpyHostToDevice ) 		);
		// cudaDeviceSynchronize(); // TODO: remove?

	}
		// std::cout << "size of m_value[0] = " << m_vStiffMat[0].num_rows() * m_vStiffMat[0].max_row_size() << std::endl;

			// cout << "gmg.cu : d_value[1] = " << endl;
			// printVector_GPU<<< 3, 1024 >>>( d_value[1], 2205 );
			// cudaDeviceSynchronize();
			// std::cout << "gmg::init" << std::endl;
			// std::cout <<"gmg.cu : d_value[0].sum()" << std::endl;
			// sum_GPU<<<9,1024>>>(d_aps, d_value[0], m_vStiffMat[0].num_rows() * m_vStiffMat[0].max_row_size() );
			// cudaDeviceSynchronize();
			// print_GPU<<<1,1>>>(d_aps);
			// cudaDeviceSynchronize();
			// std::cout <<"gmg.cu : d_value[1].sum()" << std::endl;
			// sum_GPU<<<9,1024>>>(d_aps, d_value[1], m_vStiffMat[1].num_rows() * m_vStiffMat[1].max_row_size() );
			// cudaDeviceSynchronize();
			// print_GPU<<<1,1>>>(d_aps);
			// cudaDeviceSynchronize();
			// CUDA_CALL( cudaMemcpy( d_value[1], value[1], sizeof(double)*num_rows[1]*max_row_size[1], cudaMemcpyHostToDevice ) 		);
			// CUDA_CALL( cudaMemcpy( d_gamma, &m_gamma, sizeof(int), cudaMemcpyHostToDevice ) 		);
			// CUDA_CALL( cudaMemcpy( d_topLev, &topLev, sizeof(size_t), cudaMemcpyHostToDevice ) 		);

			// cout << "d_value[0]" << endl;
			// printVector_GPU<<<1,sizeof(double)*num_rows[0]*max_row_size[0]>>>(d_value[0]);

			// cout << "d_value[0]" << endl;
			// printVector_GPU<<<1,num_rows[0]*max_row_size[0]>>>(d_value[0]);			

			// cudaDeviceSynchronize();
			// cout << "d_value[1]" << endl;
			// printVector_GPU<<<1,num_rows[1]*max_row_size[1]>>>(d_value[1]);


	return true;
}

template<std::size_t dim>
bool GMG<dim>::precond_add_update(Vector<double>& c, Vector<double>& r, std::size_t lev, int cycle) const
{
	Vector<double> ctmp(c.size(), 0.0, c.layouts());

	// for(int i = 0; i < c.size() ; i++)
	// std::cout << "ctmp[" << i << "] = " << ctmp[i]<< std::endl;

    // base solver on base level
	if(lev == m_baseLvl){
		
        if( !m_pBaseSolver->solve(ctmp, r)){
            std::cout << "Base solver failed on level " << lev << ". Aborting." << std::endl;
            return false;
        }

        c += ctmp;

		// r -= m_vStiffMat[lev] * c;
        UpdateResiduum(r, m_vStiffMat[lev], ctmp); 
        return true;
	}


	// presmooth
		// NOTE: no need to copy to device, nu1, ... ; for loop in CPU
	for(std::size_t nu1 = 0; nu1 < m_numPreSmooth; ++nu1){

		if(!m_vSmoother[lev]->precond(ctmp,r)) return false;
		ctmp.change_storage_type(PST_CONSISTENT);

		// std::cout << "precond_add_update : post-ctmp[8] = " << ctmp[8] << std::endl;
		c += ctmp;
		// r -= m_vStiffMat[lev] * ctmp;
		// std::cout << "UpdateResiduum() in precond_add_update()" << std::endl;
		UpdateResiduum(r, m_vStiffMat[lev], ctmp);
	}

	// restrict defect
	Vector<double> r_coarse(m_vProlongMat[lev-1].num_cols(), 0.0, m_multiGrid.grid(lev-1).layouts());


	// d_coarse = m_vRestrictMat[lev-1] * d;
	ApplyTransposed(r_coarse, m_vProlongMat[lev-1], r);



	// coarse grid solve
    Vector<double> c_coarse(r_coarse.size(), 0.0, r_coarse.layouts());
	
    if(cycle == _F_){
        
        // one F-Cycle ...
        if( !precond_add_update(c_coarse, r_coarse, lev-1, _F_) ){
            std::cout << "gmg failed on level " << lev << ". Aborting." << std::endl;
            return false;
        }

        // ... followed by a V-Cycle
       if( !precond_add_update(c_coarse, r_coarse, lev-1, _V_) ){
            std::cout << "gmg failed on level " << lev << ". Aborting." << std::endl;
            return false;
        }

   }
    else {
        	
        // V- and W-cycle
        for(int g = 0; g < cycle; ++g){

            if( !precond_add_update(c_coarse, r_coarse, lev-1, cycle) ){
                std::cout << "gmg failed on level " << lev << ". Aborting." << std::endl;
                return false;
            }

        }
    }

	Apply(ctmp, m_vProlongMat[lev-1], c_coarse);
	ctmp.set_storage_type(PST_CONSISTENT);

	// add correction and update defect
	c += ctmp;
	// d -= m_vStiffMat[lev] * ctmp;


	UpdateResiduum(r, m_vStiffMat[lev], ctmp);

	// postsmooth
	for(std::size_t nu2 = 0; nu2 < m_numPostSmooth; ++nu2){


		if(!m_vSmoother[lev]->precond(ctmp,r)) return false;
		ctmp.change_storage_type(PST_CONSISTENT);

		c += ctmp;

		// for(int i = 0; i < c.size() ; i++)
		// std::cout << "c[" << i << "] = " << ctmp[i]<< std::endl;

		// r -= m_vStiffMat[lev] * ctmp;
		UpdateResiduum(r, m_vStiffMat[lev], ctmp);

	}
	return true;
}


template<std::size_t dim>
bool GMG<dim>::precond(Vector<double>& c, const Vector<double>& r) const
{
	if(c.size() != r.size()){
		cout << "GMG: Size mismatch." << endl;
		return false;
	}

	std::size_t topLev = m_multiGrid.num_levels() - 1;

    // reset correction
    // c.resize(d.size()); 
    c = 0.0;
    Vector<double> rtmp(r);

	return precond_add_update(c, rtmp, topLev, m_gamma);
}


template<std::size_t dim>
bool GMG<dim>::precond_GPU(double* d_c, double* d_r)
{

	// std::cout <<"gmg.cu : precond()\n";

	std::size_t topLev = m_multiGrid.num_levels() - 1;
	// std::cout <<"gmg.cu : topLev = " << topLev << std::endl;

	// Calculating the required CUDA grid and block dimensions
	dim3 blockDim;
	dim3 gridDim;

	calculateDimensions(m_vStiffMat[topLev].num_rows(), blockDim, gridDim);

	// // DEBUG:
	// cout << "gmg.cu : precond_GPU" << endl;
	// cudaDeviceSynchronize();


	// reset correction
    // c.resize(d.size()); 
    // c = 0.0;
	setToZero<<<gridDim, blockDim>>>(d_c, m_vStiffMat[topLev].num_rows());
	// cudaDeviceSynchronize();

	// Vector<double> rtmp(r);
	vectorEquals_GPU<<<gridDim, blockDim>>>(d_rtmp[topLev], d_r, m_vStiffMat[topLev].num_rows());
	// cudaDeviceSynchronize();

	// // DEBUG:
	// cout << "GMG<dim>::precond_GPU() " << endl;
	// cout << "d_r = " << endl;
	// printVector_GPU<<< 1, m_vStiffMat[topLev].num_rows() >>>( d_r );
	// cudaDeviceSynchronize();

	// cout << "dvalue[1] = " << endl;
	// printVector_GPU<<< 3, 1024 >>>( d_value[1], m_vStiffMat[1].num_rows() * m_vStiffMat[1].max_row_size() );
	// cudaDeviceSynchronize();


	// CUDA:
	// NOTE: the original d_c and d_r from i_s.cu stay here
	// d_gmg_c[topLev] = d_c
	// d_gmg_r[topLev] = d_r
	vectorEquals_GPU<<<gridDim, blockDim>>>(d_gmg_c[topLev], d_c, m_vStiffMat[topLev].num_rows());
	// cudaDeviceSynchronize();
	vectorEquals_GPU<<<gridDim, blockDim>>>(d_gmg_r[topLev], d_r, m_vStiffMat[topLev].num_rows());
	// cudaDeviceSynchronize();

	
	// std::cout <<"gmg.cu : d_value[lev].sum()" << std::endl;
	// sum_GPU<<<3,1024>>>(d_aps, d_value[1], m_vStiffMat[1].num_rows() * m_vStiffMat[1].max_row_size() );
	// cudaDeviceSynchronize();
	// print_GPU<<<1,1>>>(d_aps);
	// cudaDeviceSynchronize();
		// 	std::cout <<"gmg.cu : d_rtmp[topLev].norm()" << std::endl;
		// norm_GPU_test(d_aps, d_rtmp[topLev], m_vStiffMat[topLev].num_rows(), gridDim, blockDim );
		// cudaDeviceSynchronize();
		// print_GPU<<<1,1>>>(d_aps);
		// cudaDeviceSynchronize();
	
	// precond_add_update(c, rtmp, topLev, m_gamma)
	// std::cout <<"gmg.cu : precond() going into pre_a_u()" << std::endl;
	// cudaDeviceSynchronize();
	precond_add_update_GPU(d_gmg_c[topLev], d_rtmp[topLev], topLev, m_gamma);	// NOTE: d_r is a dummy, not really used
	// cudaDeviceSynchronize();
	
	// std::cout << "Exiting precond_add_update_GPU" << std::endl;
	// cudaDeviceSynchronize();
	// std::cout <<"gmg.cu : d_value[lev].sum()" << std::endl;
	// sum_GPU<<<9,1024>>>(d_aps, d_value[1], m_vStiffMat[1].num_rows() * m_vStiffMat[1].max_row_size() );
	// cudaDeviceSynchronize();
	// print_GPU<<<1,1>>>(d_aps);
	// cudaDeviceSynchronize();

	// TODO:
	// make sure d_r and d_c are updated before returning to i_s.cu

	vectorEquals_GPU<<<gridDim, blockDim>>>(d_c, d_gmg_c[topLev], m_vStiffMat[topLev].num_rows());
	// cudaDeviceSynchronize();
	vectorEquals_GPU<<<gridDim, blockDim>>>(d_r, d_gmg_r[topLev], m_vStiffMat[topLev].num_rows());
	// cudaDeviceSynchronize();



	//NOTE: what's this ????
	// addVector_GPU<<<1,m_vStiffMat[topLev].num_rows()>>>(d_r, d_rtmp[topLev]);
	// cudaDeviceSynchronize();

	return true;
}


template<std::size_t dim>
bool GMG<dim>::precond_add_update_GPU(double* d_c, double* d_r, std::size_t lev, int cycle){

	// cout << "gmg.cu : precond_add_update_GPU() @ level " << lev << " $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"<< endl;
	// cudaDeviceSynchronize();	

		
		// std::cout <<"gmg.cu : d_value[lev].sum() @ level 1" << std::endl;
		// sum_GPU<<<9,1024>>>(d_aps, d_value[1], m_vStiffMat[1].num_rows() * m_vStiffMat[1].max_row_size() );
		// cudaDeviceSynchronize();
		// print_GPU<<<1,1>>>(d_aps);
		// cudaDeviceSynchronize();
		
		
		/// Calculating the required CUDA grid and block dimensions
		// Current level
		dim3 blockDim;
		dim3 gridDim;
		calculateDimensions(m_vStiffMat[lev].num_rows(), blockDim, gridDim);
		
		// Level below
		dim3 blockDim_;
		dim3 gridDim_;
		calculateDimensions(m_vProlongMat[lev-1].num_rows(), blockDim_, gridDim_);
		
		// Level below
		dim3 blockDim_cols;
		dim3 gridDim_cols;
		calculateDimensions(m_vProlongMat[lev-1].num_cols(), blockDim_cols, gridDim_cols);
		
		// std::cout <<"gmg.cu : d_r.norm()" << std::endl;
		// norm_GPU<<<gridDim, blockDim>>>(d_aps, d_r, m_vStiffMat[lev].num_rows() );
		// cudaDeviceSynchronize();
		// print_GPU<<<1,1>>>(d_aps);
		// cudaDeviceSynchronize();

		// if ( lev == 2 )
		// {
		// cout << "gmg.cu : d_r = " << endl;
		// printVector_GPU<<< gridDim, blockDim >>>( d_r, m_vStiffMat[lev].num_rows() );
		// cudaDeviceSynchronize();
		// }

		
		// std::cout <<"gmg.cu : setToZero()" << std::endl;
		// Vector<double> ctmp(c.size(), 0.0, c.layouts());
		setToZero<<< gridDim, blockDim >>>( d_ctmp[lev], m_vStiffMat[lev].num_rows() );			
		// cudaDeviceSynchronize();
		
		// std::cout <<"gmg.cu : d_value[lev].sum() @ level 1" << std::endl;
		// sum_GPU<<<9,1024>>>(d_aps, d_value[1], m_vStiffMat[1].num_rows() * m_vStiffMat[1].max_row_size() );
		// cudaDeviceSynchronize();
		// print_GPU<<<1,1>>>(d_aps);
		// cudaDeviceSynchronize();

		// // DEBUG:
		// cout << "gmg.cu : d_c = " << endl;
		// printVector_GPU<<< 1, m_vStiffMat[lev].num_rows() >>>( d_c );
		// cudaDeviceSynchronize();
		// cout << "gmg.cu : d_ctmp[lev] = " << endl;
		// printVector_GPU<<< 1, m_vStiffMat[lev].num_rows() >>>( d_ctmp[lev] );
		// cudaDeviceSynchronize();
		// cout << "gmg.cu : d_rtmp[lev] = " << endl;
		// printVector_GPU<<< 1, m_vStiffMat[lev].num_rows() >>>( d_rtmp[lev] );
		// cudaDeviceSynchronize();
		// cout << "gmg.cu : d_r = " << endl;
		// printVector_GPU<<< gridDim, blockDim >>>( d_r, m_vStiffMat[lev].num_rows() );
		// cudaDeviceSynchronize();

		// cout << "d_num_rows[lev]" << endl;
		// printVector_GPU<<< 1, 1 >>>( d_num_rows[lev] );		
		// cudaDeviceSynchronize();
		// cout << "d_max_row_size[lev]" << endl;
		// printVector_GPU<<< 1, 1 >>>( d_max_row_size[lev] );		
		// cudaDeviceSynchronize();

		
		// std::cout <<"gmg.cu : d_c.norm()" << std::endl;
		// norm_GPU<<<gridDim,blockDim>>>(d_aps, d_c, m_vStiffMat[lev].num_rows());
		// cudaDeviceSynchronize();
		// print_GPU<<<1,1>>>(d_aps);
		// cudaDeviceSynchronize();

		// std::cout <<"gmg.cu : d_r.norm()" << std::endl;
		// norm_GPU<<<gridDim, blockDim>>>(d_aps, d_r, m_vStiffMat[lev].num_rows() );
		// cudaDeviceSynchronize();
		// print_GPU<<<1,1>>>(d_aps);
		// cudaDeviceSynchronize();



		// std::cout <<"gmg.cu : d_value[lev].sum() @ level 1" << std::endl;
		// sum_GPU<<<9,1024>>>(d_aps, d_value[1], m_vStiffMat[1].num_rows() * m_vStiffMat[1].max_row_size() );
		// cudaDeviceSynchronize();
		// print_GPU<<<1,1>>>(d_aps);
		// cudaDeviceSynchronize();
	


	if(lev == m_baseLvl)
	{

		// std::cout <<"gmg.cu : d_value[lev].sum() @ level 1" << std::endl;
		// sum_GPU<<<9,1024>>>(d_aps, d_value[1], m_vStiffMat[1].num_rows() * m_vStiffMat[1].max_row_size() );
		// cudaDeviceSynchronize();
		// print_GPU<<<1,1>>>(d_aps);
		// cudaDeviceSynchronize();


			// std::cout << "gmg.cu : base solver at level 0" << std::endl;
			// cudaDeviceSynchronize();

			// std::cout <<"d_ctmp[" << lev << "]" << std::endl;
			// printVector_GPU<<<1,m_vStiffMat[lev].num_rows()>>>(d_ctmp[lev]);
			// cudaDeviceSynchronize();

			// std::cout <<"d_r" << std::endl;
			// printVector_GPU<<<1,m_vStiffMat[lev].num_rows()>>>(d_r);
			// cudaDeviceSynchronize();

        if( !m_pBaseSolver->solve_GPU(d_ctmp[lev], d_r))
		{
            std::cout << "Base solver failed on level " << lev << ". Aborting." << std::endl;
            return false;
        }

		// std::cout <<"gmg.cu : base level : After solve()" << std::endl;
		// cudaDeviceSynchronize();

		// std::cout <<"gmg.cu : d_ctmp[lev].norm()" << std::endl;
		// norm_GPU<<<gridDim,blockDim>>>(d_aps, d_ctmp[lev], m_vStiffMat[lev].num_rows());
		// cudaDeviceSynchronize();
		// print_GPU<<<1,1>>>(d_aps);
		// cudaDeviceSynchronize();

		// std::cout <<"d_ctmp[" << lev << "]" << std::endl;
        // printVector_GPU<<<1,m_vStiffMat[lev].num_rows()>>>(d_ctmp[lev]);
		// cudaDeviceSynchronize();

		// std::cout <<"d_c" << std::endl;
        // printVector_GPU<<<1,m_vStiffMat[lev].num_rows()>>>(d_c);
		// cudaDeviceSynchronize();

		
		// std::cout <<"gmg.cu : c += ctmp" << std::endl;
		// c += ctmp;
		addVector_GPU<<< gridDim, blockDim >>>(d_c, d_ctmp[lev], m_vStiffMat[lev].num_rows());


		// std::cout <<"d_c" << std::endl;
        // printVector_GPU<<<1,m_vStiffMat[lev].num_rows()>>>(d_c);
		// cudaDeviceSynchronize();

		// std::cout <<"d_ctmp[" << lev << "]" << std::endl;
        // printVector_GPU<<<1,m_vStiffMat[lev].num_rows()>>>(d_ctmp[lev]);
		// cudaDeviceSynchronize();
		
		// std::cout <<"gmg.cu : base level : UpdateResiduum" << std::endl;
		// cudaDeviceSynchronize();

		// std::cout <<"gmg.cu : d_c.norm()" << std::endl;
		// norm_GPU<<<gridDim,blockDim>>>(d_aps, d_c, m_vStiffMat[lev].num_rows());
		// cudaDeviceSynchronize();
		// print_GPU<<<1,1>>>(d_aps);
		// cudaDeviceSynchronize();
		

		// std::cout <<"gmg.cu : d_value[lev].sum() @ level 1" << std::endl;
		// sum_GPU<<<9,1024>>>(d_aps, d_value[1], m_vStiffMat[1].num_rows() * m_vStiffMat[1].max_row_size() );
		// cudaDeviceSynchronize();
		// print_GPU<<<1,1>>>(d_aps);
		// cudaDeviceSynchronize();

		

		// cout << "gmg.cu : UpdateResiduum() base solver" << endl;
		// r -= m_vStiffMat[lev] * c;
		// UpdateResiduum(r, m_vStiffMat[lev], ctmp);
		UpdateResiduum_GPU<<< gridDim, blockDim >>>(m_vStiffMat[lev].num_rows(), m_vStiffMat[lev].max_row_size(), d_value[lev], d_index[lev], d_ctmp[lev], d_r);
		// cudaDeviceSynchronize();

		// std::cout <<"gmg.cu : d_value[lev].sum() @ level 1" << std::endl;
		// sum_GPU<<<9,1024>>>(d_aps, d_value[1], m_vStiffMat[1].num_rows() * m_vStiffMat[1].max_row_size() );
		// cudaDeviceSynchronize();
		// print_GPU<<<1,1>>>(d_aps);
		// cudaDeviceSynchronize();


		// std::cout <<"d_r" << std::endl;
        // printVector_GPU<<< gridDim, blockDim >>>(d_r);
		// cudaDeviceSynchronize();	

		// std::cout <<"d_rtmp[lev]" << std::endl;
        // printVector_GPU<<<1,m_vStiffMat[lev].num_rows()>>>(d_rtmp[lev]);
		// cudaDeviceSynchronize();		

		// std::cout <<"d_ctmp[lev]" << std::endl;
        // printVector_GPU<<<1,m_vStiffMat[lev].num_rows()>>>(d_ctmp[lev]);
		// cudaDeviceSynchronize();

		// std::cout <<"gmg.cu : d_r.norm()" << std::endl;
		// norm_GPU<<<gridDim,blockDim>>>(d_aps, d_r, m_vStiffMat[lev].num_rows());
		// cudaDeviceSynchronize();
		// print_GPU<<<1,1>>>(d_aps);
		// cudaDeviceSynchronize();

		// std::cout <<"gmg.cu : base level : end of solve()" << std::endl;
		// cudaDeviceSynchronize();

        return true;
	}

	// cout << "gmg.cu : before presmooth @ level " << lev << endl;
	// cudaDeviceSynchronize();

	// cout << "d_ctmp[lev] = " << endl;
	// printVector_GPU<<< 1, m_vStiffMat[lev].num_rows() >>>( d_ctmp[lev] );
	// cudaDeviceSynchronize();

	// cout << "gmg.cu : d_rtmp[lev] = " << endl;
	// printVector_GPU<<< 1, m_vStiffMat[lev].num_rows() >>>( d_rtmp[lev] );
	// cudaDeviceSynchronize();

	// std::cout <<"gmg.cu : d_r.norm()" << std::endl;
	// norm_GPU<<<gridDim, blockDim>>>(d_aps, d_r, m_vStiffMat[lev].num_rows() );
	// cudaDeviceSynchronize();
	// print_GPU<<<1,1>>>(d_aps);
	// cudaDeviceSynchronize();

	// cout << "gmg.cu : presmooth @ level " << lev << endl;
	// cudaDeviceSynchronize();

	// presmooth
	for (std::size_t nu1 = 0; nu1 < m_numPreSmooth; ++nu1)
	{
		// precond(ctmp,r)
		// cout << "gmg.cu : inside presmooth : precond in loop " << nu1 << endl;

		// std::cout <<"gmg.cu : d_r.norm()" << std::endl;
		// norm_GPU_test(d_aps, d_r, m_vStiffMat[lev].num_rows(), gridDim, blockDim );
		// cudaDeviceSynchronize();
		// print_GPU<<<1,1>>>(d_aps);
		// cudaDeviceSynchronize();

		
		if (!m_vSmoother[lev]->precond_GPU(d_ctmp[lev], d_r)) return false;
		// cudaDeviceSynchronize();

		// std::cout <<"gmg.cu : d_r.norm()" << std::endl;
		// norm_GPU_test(d_aps, d_r, m_vStiffMat[lev].num_rows(), gridDim, blockDim );
		// cudaDeviceSynchronize();
		// print_GPU<<<1,1>>>(d_aps);
		// cudaDeviceSynchronize();

		// cout << "gmg.cu : inside presmooth : after precond " << endl;
		// cudaDeviceSynchronize();

		// 	cout << "d_ctmp[lev] = " << endl;
		// 	printVector_GPU<<< 1, m_vStiffMat[lev].num_rows() >>>( d_ctmp[lev] );
		// 	cudaDeviceSynchronize();

		// 	cout << "d_c = " << endl;
		// 	printVector_GPU<<<1, m_vStiffMat[lev].num_rows()>>>(d_c);
		// 	cudaDeviceSynchronize();

		// std::cout <<"gmg.cu : d_ctmp[lev].norm()" << std::endl;
		// norm_GPU<<<gridDim,blockDim>>>(d_aps, d_ctmp[lev], m_vStiffMat[lev].num_rows());
		// cudaDeviceSynchronize();
		// print_GPU<<<1,1>>>(d_aps);
		// cudaDeviceSynchronize();
		
		// c += ctmp;
		addVector_GPU<<< gridDim, blockDim >>>( d_c, d_ctmp[lev], m_vStiffMat[lev].num_rows() );
		// cudaDeviceSynchronize();
		
		// cout << "gmg.cu : inside presmooth : after c += ctmp" << endl;
		// cudaDeviceSynchronize();
	
		// std::cout <<"gmg.cu : d_r.norm()" << std::endl;
		// norm_GPU_test(d_aps, d_r, m_vStiffMat[lev].num_rows(), gridDim, blockDim );
		// cudaDeviceSynchronize();
		// print_GPU<<<1,1>>>(d_aps);
		// cudaDeviceSynchronize();
		
			// cout << "d_ctmp[lev] = " << endl;
			// printVector_GPU<<< 1, m_vStiffMat[lev].num_rows() >>>( d_ctmp[lev] );
			// cudaDeviceSynchronize();

			// cout << "d_gmg_c[lev] = " << endl;
			// printVector_GPU<<<1, m_vStiffMat[lev].num_rows()>>>(d_gmg_c[lev]);
			// cudaDeviceSynchronize();

			// cout << "d_r = " << endl;
			// printVector_GPU<<<gridDim, blockDim>>>(d_r, m_vStiffMat[lev].num_rows() );
			// cudaDeviceSynchronize();

			// cout << "d_ctmp[lev] = " << endl;
			// printVector_GPU<<<gridDim, blockDim>>>(d_ctmp[lev], m_vStiffMat[lev].num_rows() );
			// cudaDeviceSynchronize();

		// cout << "gmg.cu : inside presmooth : UpdateResiduum" << endl;
		// r -= m_vStiffMat[lev] * ctmp;
		// UpdateResiduum(r, m_vStiffMat[lev], ctmp);
			UpdateResiduum_GPU<<< gridDim, blockDim >>>(m_vStiffMat[lev].num_rows(), m_vStiffMat[lev].max_row_size(), d_value[lev], d_index[lev], d_ctmp[lev], d_r);
			// cudaDeviceSynchronize();

			
			// std::cout <<"gmg.cu : d_r.norm()" << std::endl;
			// norm_GPU<<<gridDim, blockDim>>>(d_aps, d_r, m_vStiffMat[lev].num_rows() );
			// cudaDeviceSynchronize();
			// print_GPU<<<1,1>>>(d_aps);
			// cudaDeviceSynchronize();

			// cout << "d_ctmp[lev] = " << endl;
			// printVector_GPU<<< 1, m_vStiffMat[lev].num_rows() >>>( d_ctmp[lev] );
			// cudaDeviceSynchronize();

			// cout << "d_c = " << endl;
			// printVector_GPU<<<1, m_vStiffMat[lev].num_rows()>>>( d_c );
			// cudaDeviceSynchronize();

			// cout << "gmg.cu : d_rtmp[lev] = " << endl;
			// printVector_GPU<<< 1, m_vStiffMat[lev].num_rows() >>>( d_rtmp[lev] );
			// cudaDeviceSynchronize();

	}

		// cout << "gmg.cu : after presmooth  " << endl;
		// cudaDeviceSynchronize();	

			// cout << "gmg.cu : d_ctmp = " << endl;
			// printVector_GPU<<< 1, m_vStiffMat[lev].num_rows() >>>( d_ctmp[lev] );
			// cudaDeviceSynchronize();

			// cout << "gmg.cu : d_c = " << endl;
			// printVector_GPU<<< 1, m_vStiffMat[lev].num_rows() >>>( d_c );
			// cudaDeviceSynchronize();

			// cout << "gmg.cu : d_rtmp[lev] = " << endl;
			// printVector_GPU<<< 1, m_vStiffMat[lev].num_rows() >>>( d_rtmp[lev] );
			// cudaDeviceSynchronize();

			// cout << "gmg.cu : restrict defect" << endl;
			// cudaDeviceSynchronize();

			// std::cout <<"gmg.cu : d_c.norm()" << std::endl;
			// norm_GPU<<<gridDim,blockDim>>>(d_aps, d_c, m_vStiffMat[lev].num_rows());
			// cudaDeviceSynchronize();
			// print_GPU<<<1,1>>>(d_aps);
			// cudaDeviceSynchronize();

		// std::cout <<"gmg.cu : d_r.norm()" << std::endl;
		// norm_GPU_test(d_aps, d_r, m_vStiffMat[lev].num_rows(), gridDim, blockDim );
		// cudaDeviceSynchronize();
		// print_GPU<<<1,1>>>(d_aps);
		// cudaDeviceSynchronize();




	// restrict defect
	// Vector<double> r_coarse(m_vProlongMat[lev-1].num_cols(), 0.0, m_multiGrid.grid(lev-1).layouts());
	
		
	setToZero<<<gridDim_cols,blockDim_cols>>>( d_gmg_r[lev-1], m_vProlongMat[lev-1].num_cols() );
	// cudaDeviceSynchronize();

		// cout << "d_rtmp[lev-1] = " << endl;
		// printVector_GPU<<< 1, m_vProlongMat[lev-1].num_cols() >>>( d_rtmp[lev-1] );
		// cudaDeviceSynchronize();

	/// r_coarse = m_vProlongMat[lev-1]^T * r
	// ApplyTransposed(r_coarse, m_vProlongMat[lev-1], r);

		// cout << "m_vProlongMat[lev-1].num_cols() = " << m_vProlongMat[lev-1].num_cols() << endl;
		// cout << "m_vProlongMat[lev-1].num_rows() = " << m_vProlongMat[lev-1].num_rows() << endl;
		// cout << "m_vProlongMat[lev-1].max_row_size() = " << m_vProlongMat[lev-1].max_row_size() << endl;

		// if(lev == 1)
		// {
		// 	cout << "d_value[1] = " << endl;
		// 	printVector_GPU<<< 3, 1024 >>>( d_value[1], 2205 );
		// 	cudaDeviceSynchronize();
		// }

		// std::cout << "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@" << std::endl;
		// std::cout <<"gmg.cu : d_value[1].sum()" << std::endl;
		// sum_GPU<<<3,1024>>>(d_aps, d_value[1], 2205 );
		// cudaDeviceSynchronize();
		// print_GPU<<<1,1>>>(d_aps);
		// cudaDeviceSynchronize();

		// cout << "gmg.cu : ApplyTransposed() @ level " << lev << endl;
		// cudaDeviceSynchronize();

	/// r_coarse = A^T * r

	// setToZero<<<gridDim_,blockDim_>>>(d_gmg_r[lev-1], m_vProlongMat[lev-1].num_rows());
	ApplyTransposed_GPU<<<gridDim_,blockDim_>>>(m_vProlongMat[lev-1].num_rows(), m_vProlongMat[lev-1].max_row_size(), d_p_value[lev-1], d_p_index[lev-1], d_r, d_gmg_r[lev-1]);
	// cudaDeviceSynchronize();


	// if(lev == 1)
	// {
	// 	cout << "d_value[1] = " << endl;
	// 	printVector_GPU<<< 3, 1024 >>>( d_value[1], 2205 );
	// 	cudaDeviceSynchronize();
	// }
	// std::cout << "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@" << std::endl;
	// std::cout <<"gmg.cu : d_value[1].sum()" << std::endl;
	// sum_GPU<<<9,1024>>>(d_aps, d_value[1], m_vStiffMat[1].num_rows() * m_vStiffMat[1].max_row_size() );
	// cudaDeviceSynchronize();
	// print_GPU<<<1,1>>>(d_aps);
	// cudaDeviceSynchronize();

		// if(lev == 1)
		// {
		// 	cout << "d_gmg_r[lev-1] = " << endl;
		// 	printVector_GPU<<< 1, m_vStiffMat[lev-1].num_rows() >>>( d_gmg_r[lev-1] );
		// 	cudaDeviceSynchronize();
		// 	cout << "d_r = " << endl;
		// 	printVector_GPU<<< 1, m_vStiffMat[lev].num_rows() >>>( d_r );
		// 	cudaDeviceSynchronize();
		// }

		// std::cout <<"gmg.cu : d_gmg_r[lev-1].norm()" << std::endl;
		// norm_GPU<<<gridDim_,blockDim_>>>(d_aps, d_gmg_r[lev-1], m_vStiffMat[lev-1].num_rows());
		// cudaDeviceSynchronize();
		// print_GPU<<<1,1>>>(d_aps);
		// cudaDeviceSynchronize();	

		// std::cout <<"gmg.cu : d_r_coarse.norm()" << std::endl;
		// norm_GPU_test(d_aps, d_gmg_r[lev-1], m_vStiffMat[lev-1].num_rows(), gridDim_, blockDim_ );
		// cudaDeviceSynchronize();
		// print_GPU<<<1,1>>>(d_aps);
		// cudaDeviceSynchronize();

		// std::cout <<"gmg.cu : d_r.norm()" << std::endl;
		// norm_GPU_test(d_aps, d_r, m_vStiffMat[lev].num_rows(), gridDim, blockDim );
		// cudaDeviceSynchronize();
		// print_GPU<<<1,1>>>(d_aps);
		// cudaDeviceSynchronize();

	// std::cout << "gmg.cu : settoZero(c_coarse)" << std::endl;
	/// coarse grid solve
	// Vector<double> c_coarse(r_coarse.size(), 0.0, r_coarse.layouts());
	setToZero<<<gridDim_cols,blockDim_cols>>>( d_gmg_c[lev-1], m_vProlongMat[lev-1].num_cols() );
	// cudaDeviceSynchronize();

		// cout << "d_gmg_c[lev-1] = " << endl;
		// printVector_GPU<<< 1, m_vStiffMat[lev-1].num_rows() >>>( d_gmg_c[lev-1] );
		// cudaDeviceSynchronize();

		// 	std::cout <<"gmg.cu : d_r_coarse.norm()" << std::endl;
		// norm_GPU_test(d_aps, d_gmg_r[lev-1], m_vStiffMat[lev-1].num_rows(), gridDim_, blockDim_ );
		// cudaDeviceSynchronize();
		// print_GPU<<<1,1>>>(d_aps);
		// cudaDeviceSynchronize();



	// cout << "gmg.cu : Before cycles @ level " << lev << endl;
	// cudaDeviceSynchronize();

	// std::cout << "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@" << std::endl;
			// std::cout <<"gmg.cu : d_value[1].sum()" << std::endl;
			// sum_GPU<<<9,1024>>>(d_aps, d_value[1], m_vStiffMat[1].num_rows() * m_vStiffMat[1].max_row_size() );
			// cudaDeviceSynchronize();
			// print_GPU<<<1,1>>>(d_aps);
			// cudaDeviceSynchronize();

		// std::cout <<"gmg.cu : d_r.norm()" << std::endl;
		// norm_GPU_test(d_aps, d_r, m_vStiffMat[lev].num_rows(), gridDim, blockDim );
		// cudaDeviceSynchronize();
		// print_GPU<<<1,1>>>(d_aps);
		// cudaDeviceSynchronize();

		// 		std::cout <<"gmg.cu : d_r_coarse.norm()" << std::endl;
		// norm_GPU_test(d_aps, d_gmg_r[lev-1], m_vStiffMat[lev-1].num_rows(), gridDim_, blockDim_ );
		// cudaDeviceSynchronize();
		// print_GPU<<<1,1>>>(d_aps);
		// cudaDeviceSynchronize();

	if(cycle == _F_)
	{

		// cout << "F cycle" << endl; // DEBUG:
		// cudaDeviceSynchronize();
			// one F-Cycle ...
		    if( !precond_add_update_GPU(d_ctmp[lev-1], d_rtmp[lev-1], lev-1, _F_) )
			{
		        std::cout << "gmg failed on level " << lev << ". Aborting." << std::endl;
		        return false;
		    }

		    // ... followed by a V-Cycle
		   	if( !precond_add_update_GPU(d_ctmp[lev-1], d_rtmp[lev-1], lev-1, _V_) )
			{
		        std::cout << "gmg failed on level " << lev << ". Aborting." << std::endl;
		        return false;
			}
	}

	else
	{
		// cout << "V or W cycle" << endl; // DEBUG:
		// cudaDeviceSynchronize();
	
		// V- and W-cycle
		for (int g = 0; g < cycle; ++g)
		{

			if( !precond_add_update_GPU(d_gmg_c[lev-1], d_gmg_r[lev-1], lev-1, cycle) )
			{
				std::cout << "gmg failed on level " << lev << ". Aborting." << std::endl;
				return false;
			}
		
		//DEBUG:

		// cout << "in V cycle, d_ctmp[lev-1] = " << endl;
		// printVector_GPU<<< 1, m_vStiffMat[lev-1].num_rows() >>>( d_ctmp[lev-1] );
		// cudaDeviceSynchronize();

			
		}
	}

	// cout << "gmg.cu : After cycles @ level " << lev << endl;
	// cudaDeviceSynchronize();

			// std::cout << "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@" << std::endl;
			// std::cout <<"gmg.cu : d_value[1].sum()" << std::endl;
			// sum_GPU<<<9,1024>>>(d_aps, d_value[1], m_vStiffMat[1].num_rows() * m_vStiffMat[1].max_row_size() );
			// cudaDeviceSynchronize();
			// print_GPU<<<1,1>>>(d_aps);
			// cudaDeviceSynchronize();

	// std::cout <<"gmg.cu : d_gmg_c[lev-1].norm()" << std::endl;
	// norm_GPU<<<gridDim_,blockDim_>>>(d_aps, d_gmg_c[lev-1], m_vStiffMat[lev-1].num_rows());
	// cudaDeviceSynchronize();
	// print_GPU<<<1,1>>>(d_aps);
	// cudaDeviceSynchronize();

	// std::cout <<"gmg.cu : d_gmg_r[lev-1].norm()" << std::endl;
	// norm_GPU<<<gridDim_,blockDim_>>>(d_aps, d_gmg_r[lev-1], m_vStiffMat[lev-1].num_rows());
	// cudaDeviceSynchronize();
	// print_GPU<<<1,1>>>(d_aps);
	// cudaDeviceSynchronize();

	// cout << "d_gmg_c[lev-1] = " << endl;
	// printVector_GPU<<< 1, m_vStiffMat[lev-1].num_rows() >>>( d_gmg_c[lev-1] );
	// cudaDeviceSynchronize();

	// cout << "d_c = " << endl;
	// printVector_GPU<<< 1, m_vStiffMat[lev].num_rows() >>>( d_c );
	// cudaDeviceSynchronize();

	// cout << "gmg.cu : ApplyGPU @ level " << lev << endl;
	// cudaDeviceSynchronize();

	/// prolongate coarse grid correction
	// ctmp = m_vProlongMat[lev-1] * c_coarse;
	// Apply(ctmp, m_vProlongMat[lev-1], c_coarse);
	Apply_GPU<<<gridDim_,blockDim_>>>( m_vProlongMat[lev-1].num_rows(), m_vProlongMat[lev-1].max_row_size(), d_p_value[lev-1], d_p_index[lev-1], d_gmg_c[lev-1], d_ctmp[lev]);
	// cudaDeviceSynchronize();

	// cout << "gmg.cu : after ApplyGPU @ level " << lev << endl;
	// cudaDeviceSynchronize();

	// std::cout <<"gmg.cu : d_ctmp[lev].norm()" << std::endl;
	// norm_GPU<<<gridDim,blockDim>>>(d_aps, d_ctmp[lev], m_vStiffMat[lev].num_rows());
	// cudaDeviceSynchronize();
	// print_GPU<<<1,1>>>(d_aps);
	// cudaDeviceSynchronize();

	// cout << "d_c = " << endl;
	// printVector_GPU<<< 1, m_vStiffMat[lev].num_rows() >>>( d_c );
	// cudaDeviceSynchronize();

	// cout << "d_ctmp = " << endl;
	// printVector_GPU<<< 1, m_vStiffMat[lev].num_rows() >>>( d_ctmp[lev] );
	// cudaDeviceSynchronize();

	// cout << "gmg.cu : add correction and update defect @ level " << lev << endl;
	// cudaDeviceSynchronize();

	//TODO:
	/// add correction and update defect
	// c += ctmp;
	addVector_GPU<<<gridDim,blockDim>>>(d_c, d_ctmp[lev], m_vStiffMat[lev].num_rows());
	// cudaDeviceSynchronize();


	// cout << "c += ctmp" << endl;
	// cout << "d_c = " << endl;
	// printVector_GPU<<< 1, m_vStiffMat[lev].num_rows() >>>( d_c );
	// cudaDeviceSynchronize();

	// std::cout <<"gmg.cu : d_c.norm()" << std::endl;
	// norm_GPU<<<gridDim,blockDim>>>(d_aps, d_c, m_vStiffMat[lev].num_rows());
	// cudaDeviceSynchronize();
	// print_GPU<<<1,1>>>(d_aps);
	// cudaDeviceSynchronize();

	/// d -= m_vStiffMat[lev] * ctmp;
	// UpdateResiduum(r, m_vStiffMat[lev], ctmp);

	// std::cout <<"gmg.cu : d_r.norm()" << std::endl;
	// norm_GPU<<<gridDim,blockDim>>>(d_aps, d_r, m_vStiffMat[lev].num_rows());
	// cudaDeviceSynchronize();
	// print_GPU<<<1,1>>>(d_aps);
	// cudaDeviceSynchronize();

	
	// std::cout <<"gmg.cu : #############################################" << std::endl;

	//TODO:
	// CUDA_CALL(cudaMemcpy(d_value[1], m_vStiffMat[1].getValueAddress(), sizeof(double) * m_vStiffMat[1].num_rows() * m_vStiffMat[1].max_row_size(), cudaMemcpyHostToDevice));
	// cudaDeviceSynchronize();
	
	
	// std::cout <<"gmg.cu : d_value[1].sum()" << std::endl;
	// sum_GPU<<<9,1024>>>(d_aps, d_value[1], m_vStiffMat[1].num_rows() * m_vStiffMat[1].max_row_size() );
	// cudaDeviceSynchronize();
	// print_GPU<<<1,1>>>(d_aps);
	// cudaDeviceSynchronize();





	// cout << "gmg.cu : UpdateResiduum_GPU @ level " << lev << endl;
	// cudaDeviceSynchronize();

	UpdateResiduum_GPU<<<gridDim,blockDim>>>( m_vStiffMat[lev].num_rows(), m_vStiffMat[lev].max_row_size() , d_value[lev], d_index[lev], d_ctmp[lev], d_r);
	// cudaDeviceSynchronize();

	// cout << "gmg.cu : After UpdateResiduum_GPU " << endl;

	// cout << "d_r = " << endl;
	// printVector_GPU<<<gridDim,blockDim>>>( d_r );
	// cudaDeviceSynchronize();

	// cout << "d_rtmp[lev] = " << endl;
	// printVector_GPU<<< 1, m_vStiffMat[lev].num_rows() >>>( d_rtmp[lev] );
	// cudaDeviceSynchronize();

	// vectorEquals_GPU<<<1, m_vStiffMat[lev].num_rows() >>> (d_c, d_ctmp[lev]);

	// cout << "d_c = " << endl;
	// printVector_GPU<<< 1, m_vStiffMat[lev].num_rows() >>>( d_c );
	// cudaDeviceSynchronize();

	// cout << "d_ctmp[lev] = " << endl;
	// printVector_GPU<<< 1, m_vStiffMat[lev].num_rows() >>>( d_ctmp[lev] );
	// cudaDeviceSynchronize();

	// std::cout <<"gmg.cu : d_r.norm()" << std::endl;
	// norm_GPU<<<gridDim,blockDim>>>(d_aps, d_r, m_vStiffMat[lev].num_rows());
	// cudaDeviceSynchronize();
	// print_GPU<<<1,1>>>(d_aps);
	// cudaDeviceSynchronize();


	// cout << "gmg.cu : Before postsmooth @ level " << lev << endl;
	// cudaDeviceSynchronize();

	// postsmooth
	for(std::size_t nu2 = 0; nu2 < m_numPostSmooth; ++nu2)
	{
		// cout << "gmg.cu : inside postsmooth" << endl;
		if(!m_vSmoother[lev]->precond_GPU(d_ctmp[lev],d_r)) return false;
		// cudaDeviceSynchronize();	

		// c += ctmp;
		addVector_GPU<<<gridDim,blockDim>>>(d_c, d_ctmp[lev], m_vStiffMat[lev].num_rows());
		// cudaDeviceSynchronize();
		
		/// r -= m_vStiffMat[lev] * ctmp;
		// UpdateResiduum(r, m_vStiffMat[lev], ctmp);
		UpdateResiduum_GPU<<<gridDim,blockDim>>>(m_vStiffMat[lev].num_rows(), m_vStiffMat[lev].max_row_size(), d_value[lev], d_index[lev], d_ctmp[lev], d_r);
		// cudaDeviceSynchronize();

	}

	// cout << "gmg.cu : After Postsmooth on level " << lev << endl;

	// cout << "d_rtmp[lev] = " << endl;
	// printVector_GPU<<< 1, m_vStiffMat[lev].num_rows() >>>( d_rtmp[lev] );
	// cudaDeviceSynchronize();

	// vectorEquals_GPU<<<1, m_vStiffMat[lev].num_rows() >>> (d_c, d_ctmp[lev]){

		// cout << "d_ctmp[lev] = " << endl;
		// printVector_GPU<<<gridDim,blockDim>>>( d_ctmp[lev] );
		// cudaDeviceSynchronize();

	// cout << "d_c = " << endl;
	// printVector_GPU<<< 1, m_vStiffMat[lev].num_rows() >>>( d_c );
	// cudaDeviceSynchronize();

	// std::cout <<"gmg.cu : d_c.norm()" << std::endl;
	// norm_GPU<<<gridDim,blockDim>>>(d_aps, d_c, m_vStiffMat[lev].num_rows());
	// cudaDeviceSynchronize();
	// print_GPU<<<1,1>>>(d_aps);
	// cudaDeviceSynchronize();

	// cout << "gmg.cu : end of while(foo) at level " << lev << endl;
	// cudaDeviceSynchronize();
	
	return true;

}	


// explicit template declarations
template class GMG<1>;
template class GMG<2>;
template class GMG<3>;