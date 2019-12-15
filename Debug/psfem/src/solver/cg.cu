/*
 * cg.cpp
 *
 * author: a.vogel@rub.de
 *
 */

#include "cg.h"
#include "parallel/parallel.h"
#include "algebra/vector.h"
#include <cassert>
#include <iostream>
#include <iomanip> // setw
#include <cmath> //sqrt

using namespace std;


////////////////////////////////////////
// CG
////////////////////////////////////////


CG::CG()
: m_pLinearIterator(0), m_maxIter(10), m_minRes(1e-50), m_minRed(1e-10), m_bVerbose(true)
{}


CG::CG(const ELLMatrix<double>& mat)
: m_pLinearIterator(0), m_maxIter(10), m_minRes(1e-50), m_minRed(1e-10), m_bVerbose(true)
{
	
	init(mat);
}


void CG::set_linear_iterator(LinearIterator& linIter)
{
	m_pLinearIterator = &linIter;
	if(this->m_pA)
		m_pLinearIterator->init(*this->m_pA);
}


bool CG::init(const ELLMatrix<double>& A)
{
	
	this->m_pA = &A;

	std::vector<double> p(A.num_rows(), 0);
	std::vector<double> r(A.num_rows(), 0);
	std::vector<double> z(A.num_rows(), 0);

	const auto value = A.getValueAddress();
	const auto index = A.getIndexAddress();
	const size_t num_rows = A.num_rows();
	const size_t max_row_size = A.max_row_size();

	// Allocate memory in device
			
	CUDA_CALL( cudaMalloc( (void**)&d_cg_p, sizeof(double)*p.size() ) 	);
	CUDA_CALL( cudaMalloc( (void**)&d_cg_r, sizeof(double)*r.size() ) 	);
	CUDA_CALL( cudaMalloc( (void**)&d_cg_z, sizeof(double)*z.size() ) 	);
	CUDA_CALL( cudaMalloc( (void**)&d_cg_res0, sizeof(double) ) 		);
	CUDA_CALL( cudaMalloc( (void**)&d_cg_res, sizeof(double) ) 		);
	CUDA_CALL( cudaMalloc( (void**)&d_cg_lastRes, sizeof(double) ) 	);
	CUDA_CALL( cudaMalloc( (void**)&d_cg_m_minRes, sizeof(double) ) 	);
	CUDA_CALL( cudaMalloc( (void**)&d_cg_m_minRed, sizeof(double) ) 	);
	CUDA_CALL( cudaMalloc( (void**)&d_cg_step, sizeof(size_t) ) 		);
	CUDA_CALL( cudaMalloc( (void**)&d_cg_m_maxIter, sizeof(size_t) ) 	);
	CUDA_CALL( cudaMalloc( (void**)&d_cg_foo, sizeof(bool) ) 			);
	CUDA_CALL( cudaMalloc( (void**)&d_cg_value, A.max_row_size() * A.num_rows() * sizeof(double) ) 		);
	CUDA_CALL( cudaMalloc( (void**)&d_cg_index, A.max_row_size() * A.num_rows() * sizeof(size_t) ) 	);
	CUDA_CALL( cudaMalloc( (void**)&d_cg_rho, sizeof(double) ) 		);
	CUDA_CALL( cudaMalloc( (void**)&d_cg_rho_old, sizeof(double) ) 		);
	CUDA_CALL( cudaMalloc( (void**)&d_cg_alpha, sizeof(double) ) 		);
	CUDA_CALL( cudaMalloc( (void**)&d_cg_alpha_temp, sizeof(double) ) 		);

// Copy memory to device

	// CUDA_CALL( cudaMemcpy( d_cg_x, &x[0], x.size() * sizeof(double), cudaMemcpyHostToDevice) 	);
	// CUDA_CALL( cudaMemcpy( d_cg_b, &b[0], b.size() * sizeof(double), cudaMemcpyHostToDevice) 	);
	CUDA_CALL( cudaMemcpy( d_cg_p, &p[0], p.size() * sizeof(double), cudaMemcpyHostToDevice) 	);
	CUDA_CALL( cudaMemcpy( d_cg_r, &r[0], r.size() * sizeof(double), cudaMemcpyHostToDevice) 	);
	CUDA_CALL( cudaMemcpy( d_cg_z, &z[0], z.size() * sizeof(double), cudaMemcpyHostToDevice) 	);
	CUDA_CALL( cudaMemcpy( d_cg_res0, &res0, sizeof(double), cudaMemcpyHostToDevice) 			);
	CUDA_CALL( cudaMemcpy( d_cg_res, &res, sizeof(double), cudaMemcpyHostToDevice) 			);
	CUDA_CALL( cudaMemcpy( d_cg_lastRes, &lastRes, sizeof(double), cudaMemcpyHostToDevice) 	);
	CUDA_CALL( cudaMemcpy( d_cg_m_minRes, &m_minRes, sizeof(double), cudaMemcpyHostToDevice) 	);
	CUDA_CALL( cudaMemcpy( d_cg_m_minRed, &m_minRed, sizeof(double), cudaMemcpyHostToDevice) 	);
	CUDA_CALL( cudaMemcpy( d_cg_step, &step, sizeof(size_t), cudaMemcpyHostToDevice) 			);
	CUDA_CALL( cudaMemcpy( d_cg_m_maxIter, &m_maxIter, sizeof(size_t), cudaMemcpyHostToDevice)	);
	CUDA_CALL( cudaMemcpy( d_cg_foo, &foo, sizeof(bool), cudaMemcpyHostToDevice) 				);
	CUDA_CALL( cudaMemcpy( d_cg_value, value, A.max_row_size() * A.num_rows() * sizeof(double), cudaMemcpyHostToDevice) 		);
	CUDA_CALL( cudaMemcpy( d_cg_index, index, A.max_row_size() * A.num_rows() * sizeof(size_t), cudaMemcpyHostToDevice) 	);
	CUDA_CALL( cudaMemcpy( d_cg_rho, &rho, sizeof(double), cudaMemcpyHostToDevice) 			);
	CUDA_CALL( cudaMemcpy( d_cg_rho_old, &rho_old, sizeof(double), cudaMemcpyHostToDevice) 	);

	// Calculating the required CUDA grid and block dimensions
	calculateDimensions(A.num_rows(), blockDim, gridDim);

	

	if(m_pLinearIterator)
		return m_pLinearIterator->init(A);
	else
		return true;
}


void CG::set_convergence_params
(
	size_t maxIter,
	double minRes,
	double minRed
)
{
	m_maxIter = maxIter;
	m_minRes = minRes;
	m_minRed = minRed;
}


void CG::set_verbose(bool verbose)
{
	m_bVerbose = verbose;
}

// NOTE: solve with d_x and d_b
bool CG::solve_GPU(double* d_x, double* d_b)
{
	// cout << "cg.cu : solve_GPU()" << endl;
	// cudaDeviceSynchronize();	
	
	// check for matrix and get reference
	if(!this->m_pA){
		cout << "Matrix has not been set via set_matrix()." << endl;
		return false;
	}

	const ELLMatrix<double>& A = *(this->m_pA);

	setToZero<<<gridDim,blockDim>>>(d_cg_p, A.num_rows());
	setToZero<<<gridDim,blockDim>>>(d_cg_r, A.num_rows());
	setToZero<<<gridDim,blockDim>>>(d_cg_z, A.num_rows());
	setToZero<<<1,1>>>( d_cg_step, 1);
	setToZero<<<1,1>>>( d_cg_rho, 1 );
	setToZero<<<1,1>>>( d_cg_rho_old, 1 );
	setToZero<<<1,1>>>( d_cg_res0, 1 );
	setToZero<<<1,1>>>( d_cg_res, 1 );
	setToZero<<<1,1>>>( d_cg_lastRes, 1 );
	

	//TODO: to delete
	// setToTrue<<<1,1>>>( d_cg_foo );
	

	// [CUDA] ---------------------------------------------------------------------------
			
		

	

	// // DEBUG: checking that the vectors are not empty
	// std::cout << "cg.cu : Before ComputeResiduum" << std::endl;
	
		// std::cout <<"i_s.cu : d_cg_r.norm()" << std::endl;
		// norm_GPU<<<gridDim,blockDim>>>(d_cg_res, d_cg_r, num_rows);
		// cudaDeviceSynchronize();
		// print_GPU<<<1,1>>>(d_cg_res);
		// cudaDeviceSynchronize();
	// std::cout << "cg.cu : d_x = " << std::endl;
	// printVector_GPU<<<1, A.num_rows()>>>(d_x);
	// cudaDeviceSynchronize();

	// std::cout << "cg.cu : d_b = " << std::endl;
	// printVector_GPU<<<1, A.num_rows()>>>(d_b);
	// cudaDeviceSynchronize();
	// std::cout << "cg.cu : d_cg_r = " << std::endl;
	// printVector_GPU<<<1, A.num_rows()>>>(d_cg_r);
	// cudaDeviceSynchronize();

	// std::cout << "cg.cu : ComputeResiduum" << std::endl;

	// compute residuum: r = b-Ax
	// ComputeResiduum(r, b, A, x);
	ComputeResiduum_GPU<<<gridDim,blockDim>>>(A.num_rows(), A.max_row_size(), d_cg_value, d_cg_index, d_x, d_cg_r, d_b);
	// cudaDeviceSynchronize();

	// std::cout <<"i_s.cu : d_cg_r.norm()" << std::endl;
	// norm_GPU<<<gridDim,blockDim>>>(d_cg_res, d_cg_r, num_rows);
	// cudaDeviceSynchronize();
	// print_GPU<<<1,1>>>(d_cg_res);
	// cudaDeviceSynchronize();

	// std::cout << "d_cg_r = " << std::endl;
	// printVector_GPU<<<gridDim,blockDim>>>(d_cg_r, A.num_rows());
	// cudaDeviceSynchronize();	

	// std::cout << "cg.cu : d_x = " << std::endl;
	// printVector_GPU<<<1, A.num_rows()>>>(d_x);
	// cudaDeviceSynchronize();

	// std::cout << "cg.cu : d_b = " << std::endl;
	// printVector_GPU<<<1, A.num_rows()>>>(d_b);
	// cudaDeviceSynchronize();


	// std::cout << "cg.cu : r.norm()" << std::endl;
	// cudaDeviceSynchronize();

	// initial residuum
	// double res0, res, lastRes; 
	// res0 = res = r.norm();
	// norm_GPU<<<gridDim,blockDim>>>(d_cg_res, d_cg_r, A.num_rows());
	//TODO:
	norm_GPU(d_cg_res, d_cg_r, A.num_rows(), gridDim, blockDim);
	// cudaDeviceSynchronize();
	equals_GPU<<<1,1>>>(d_cg_res0, d_cg_res);
	// cudaDeviceSynchronize();
	// norm_GPU<<<gridDim,blockDim>>>(d_cg_res0, d_cg_r, A.num_rows());
	// cudaDeviceSynchronize();

	// cout << "cg.cu : d_cg_res = " << endl;
	// print_GPU<<<1, 1>>>(d_cg_res);
	// cudaDeviceSynchronize();

	// step counter
	// size_t step = 0; NOTE:done above

	// help values
	// double rho, rho_old; NOTE: done above
	// cudaDeviceSynchronize();



	
	// TODO: don't display when gmg
	// output initial values
	if (m_bVerbose && mpi::world().rank() == 0)
	{
		cout << "## CG  ##################################################################" << endl;
		cout << "  Iter     Residuum       Required       Rate        Reduction     Required" << endl;
	}

	if (m_bVerbose == 1)
	{
		printInitialResult_GPU<<<1,1>>>(d_cg_res0, d_cg_m_minRes, d_cg_m_minRed);
		cudaDeviceSynchronize();
	}

	//NOTE:
		// if( *res < *m_minRes )
		// 	*foo = false;

		// if( *res < (*m_minRed)*(*res0) )
		// 	*foo = false;

		// if( *step > *m_maxIter)
		// 	*foo = false;

	addStep<<<1,1>>>(d_cg_step);
	// cudaDeviceSynchronize();

	checkIterationConditions<<<1,1>>>(d_cg_foo, d_cg_step, d_cg_res, d_cg_res0, d_cg_m_minRes, d_cg_m_minRed, d_cg_m_maxIter);
	// cudaDeviceSynchronize();

	CUDA_CALL( cudaMemcpy( &foo, d_cg_foo, sizeof(bool), cudaMemcpyDeviceToHost) 	);
	// cudaDeviceSynchronize();
	

	// cout << "cg.cu : before while(foo)" << endl;	
	// cout << "cg.cu : host : foo = " << foo << endl;


	// // res > m_minRes && res > m_minRed*res0 && ++step <= m_maxIter
	// cout << "d_cg_res > d_cg_m_minRes" << endl;
	// cout << "d_cg_res > d_cg_m_minRed*d_cg_res0" << endl;
	// cout << "++step <= m_maxIter" << endl;
	// // d_cg_foo, d_cg_step, 
	// // d_cg_res, d_cg_res0, 
	// // d_cg_m_minRes, d_cg_m_minRed, 
	// // d_cg_m_maxIter

	// // if( *res < *m_minRes )
	// // 	*foo = false;

	// cout << "cg.cu : d_cg_res = ";
	// print_GPU<<<1,1>>>(d_cg_res); // DEBUG:
	// cudaDeviceSynchronize();

	// cout << "cg.cu : d_cg_m_minRes = ";
	// print_GPU<<<1,1>>>(d_cg_m_minRes); // DEBUG:
	// cudaDeviceSynchronize();

	// // if( *res < (*m_minRed)*(*res0) )
	// // 	*foo = false;

	// cout << "cg.cu : d_cg_m_minRed = ";
	// print_GPU<<<1,1>>>(d_cg_m_minRed); // DEBUG:
	// cudaDeviceSynchronize();

	// cout << "cg.cu : d_cg_res0 = ";
	// print_GPU<<<1,1>>>(d_cg_res0); // DEBUG:
	// cudaDeviceSynchronize();

	// // if( *step > *m_maxIter)
	// // 	*foo = false;

	// cout << "cg.cu : d_cg_step = ";
	// print_GPU<<<1,1>>>(d_cg_step); // DEBUG:
	// cudaDeviceSynchronize();

	// cout << "cg.cu : d_cg_m_maxIter = ";
	// print_GPU<<<1,1>>>(d_cg_m_maxIter); // DEBUG:
	// cudaDeviceSynchronize();

	// iteration
	while (foo)
	{
		
		// cout << "cg.cu : starting while(foo)" << endl;

		// std::cout << "d_cg_step = " << std::endl;
		// print_GPU<<<1,1>>> ( d_cg_step );
		// cudaDeviceSynchronize();

			// std::cout << "d_cg_z = " << std::endl;
			// printVector_GPU<<<1, A.num_rows()>>>(d_cg_z);
			// cudaDeviceSynchronize();
		
			// std::cout << "d_cg_r = " << std::endl;
			// printVector_GPU<<<1, A.num_rows()>>>(d_cg_r);
			// cudaDeviceSynchronize();

		// cout << "cg.cu : precond()" << endl;

		// apply preconditioner
		if(m_pLinearIterator){
			if (!m_pLinearIterator->precond_GPU(d_cg_z, d_cg_r)){
				cout << "CG failed. Step method failure in step " << step << "." << endl;
				return false;
			}
		}
		else{
			// z = r;
			// cout << "cg.cu : precond() pundek : z = r" << endl;
			// cout << "pundek" << endl;
			vectorEquals_GPU<<<gridDim,blockDim>>>(d_cg_z, d_cg_r, A.num_rows());
			// cudaDeviceSynchronize();
		}
		// cout << "cg.cu : after precond()" << endl;

		// cout << " d_cg_z" << endl;
		// printVector_GPU<<<1,A.num_rows()>>>(d_cg_z);
		// cudaDeviceSynchronize();

		// cout << " d_cg_r" << endl;
		// printVector_GPU<<<1,A.num_rows()>>>(d_cg_r);
		// cudaDeviceSynchronize();
		
		// force z to be consistent
		// if (!z.change_storage_type(PST_CONSISTENT)){
		// 	cout << "Failed to change storage type for z." << endl;
		// 	return false;
		// }
		
		// cout << "cg.cu : dotproduct for rho = " << endl;
		// rho = (z,r)
		// rho = r * z;
		// dotProduct<<<gridDim,blockDim>>>(d_cg_rho, d_cg_r, d_cg_z, A.num_rows());
		dotProduct_test(d_cg_rho, d_cg_r, d_cg_z, A.num_rows(), gridDim, blockDim);
		// cudaDeviceSynchronize();

		// cout << "cg.cu : d_cg_rho = ";
		// print_GPU<<<1,1>>>(d_cg_rho); // DEBUG:
		// cudaDeviceSynchronize();

		// cout << "d_cg_rho_old = ";
		// print_GPU<<<1,1>>>(d_cg_rho_old); // DEBUG:
		// cudaDeviceSynchronize();

		// cout << "cg.cu : d_cg_step = ";
		// print_GPU<<<1,1>>>(d_cg_step); // DEBUG:
		// cudaDeviceSynchronize();

		// if(step == 1) p = z;
		// else {
		// 	p *= (rho / rho_old);
		// 	p += z;
		// }


		
		// cout << " d_cg_rho" << endl;
		// print_GPU<<<1,1>>>(d_cg_rho); // DEBUG:
		// cudaDeviceSynchronize();

		// cout << " d_cg_rho_old" << endl;
		// print_GPU<<<1,1>>>(d_cg_rho_old); // DEBUG:
		// cudaDeviceSynchronize();

		// cout << " d_cg_step" << endl;
		// print_GPU<<<1,1>>>(d_cg_step); // DEBUG:
		// cudaDeviceSynchronize();

		// cout << "cg.cu : calling calculateDirectionVector<<<>>>()" << endl;
		calculateDirectionVector<<<gridDim,blockDim>>>(d_cg_step, d_cg_p, d_cg_z, d_cg_rho, d_cg_rho_old, A.num_rows());
		// cudaDeviceSynchronize();
						
		// cout << "cg.cu : after p += z " << endl;
		// cudaDeviceSynchronize();

		// cout << " d_cg_p" << endl;
		// printVector_GPU<<<1,A.num_rows()>>>(d_cg_p);
		// cudaDeviceSynchronize();
		

		// cout << "cg.cu : before apply()" << endl;
		// cudaDeviceSynchronize();

		// cout << " d_cg_z" << endl;
		// printVector_GPU<<<1,A.num_rows()>>>(d_cg_z);
		// cudaDeviceSynchronize();

		/// z = A*p
		// Apply(z, A, p);
		// cout << "cg.cu : calling Apply_GPU<<<>>>()" << endl;
		Apply_GPU<<<gridDim,blockDim>>>( A.num_rows(), A.max_row_size(), d_cg_value, d_cg_index, d_cg_p, d_cg_z );
		// cudaDeviceSynchronize();
		// cout << "cg.cu : after apply()" << endl;

		// //DEBUG:
		// cout << " d_cg_z" << endl;
		// printVector_GPU<<<1,A.num_rows()>>>(d_cg_z);
		// cudaDeviceSynchronize();


		// cout << "cg.cu : calculateAlpha()" << endl;
		// alpha = rho / (p * z)
		// calculateAlpha<<<gridDim,blockDim>>>( d_cg_alpha, d_cg_rho, d_cg_p, d_cg_z, d_cg_alpha_temp, A.num_rows() );
		//TODO:
		calculateAlpha_test(d_cg_alpha, d_cg_rho, d_cg_p, d_cg_z, d_cg_alpha_temp, A.num_rows(), gridDim, blockDim );
		// cudaDeviceSynchronize();

		// cout << "cg.cu : d_cg_alpha : " << endl;
		// print_GPU<<<1,1>>>(d_cg_alpha); // DEBUG:
		// cudaDeviceSynchronize();
		
		// cout << "cg.cu : add correction to solution" << endl;
		// cout << "cg.cu : x = x + alpha * p" << endl;
		// add correction to solution
		// x = x + alpha * p
		axpy_GPU<<<gridDim,blockDim>>>(d_x, d_cg_alpha, d_cg_p, A.num_rows());
		// cudaDeviceSynchronize();

		// //DEBUG:
		// cout << " d_x" << endl;
		// printVector_GPU<<<1,A.num_rows()>>>(d_x);
		// cudaDeviceSynchronize();
		// //DEBUG:
		// cout << "cg.cu : d_cg_z" << endl;
		// printVector_GPU<<<1,A.num_rows()>>>(d_cg_z);
		// cudaDeviceSynchronize();

		//NOTE: okay till here

		// cout << "cg.cu : update residuum" << endl;
		// update residuum
		// r = r - alpha * z
		axpy_neg_GPU<<<gridDim,blockDim>>>(d_cg_r, d_cg_alpha, d_cg_z, A.num_rows());
		// cudaDeviceSynchronize();

		//DEBUG:
		// cout << "cg.cu : axpy_neg_GPU" << endl;
		// cudaDeviceSynchronize();
		// printVector_GPU<<<1,A.num_rows()>>>(d_cg_r);
		// cudaDeviceSynchronize();


		
		// cout << "cg.cu : compute residuum" << endl;
		// compute residuum
		// lastRes = res;
		equals_GPU<<<1,1>>>(d_cg_lastRes, d_cg_res);
		// cudaDeviceSynchronize();

		// cout << "cg.cu : compute norm()" << endl;
		// res = r.norm();
		// norm_GPU<<<gridDim,blockDim>>>(d_cg_res, d_cg_r, A.num_rows());
		//TODO:
		norm_GPU(d_cg_res, d_cg_r, A.num_rows(), gridDim, blockDim);
		// cudaDeviceSynchronize();

		// cout << "cg.cu : r.norm()" << endl;
		// print_GPU<<<1,1>>>(d_cg_res);

		// store old rho
		// rho_old = rho;
		vectorEquals_GPU<<<gridDim,blockDim>>>(d_cg_rho_old, d_cg_rho, A.num_rows());
		// cudaDeviceSynchronize();

		if (m_bVerbose == 1)
		{
			// output iteration progress
			printResult_GPU<<<1,1>>>(d_cg_step, d_cg_res, d_cg_m_minRes, d_cg_lastRes, d_cg_res0, d_cg_m_minRed);
			cudaDeviceSynchronize();
		}

		addStep<<<1,1>>>(d_cg_step);
		// cudaDeviceSynchronize();
		
		checkIterationConditions<<<1,1>>>(d_cg_foo, d_cg_step, d_cg_res, d_cg_res0, d_cg_m_minRes, d_cg_m_minRed, d_cg_m_maxIter);
		// cudaDeviceSynchronize();


		CUDA_CALL( cudaMemcpy( &foo, d_cg_foo, sizeof(bool), cudaMemcpyDeviceToHost) 	);
		// cudaDeviceSynchronize();

		// // //DEBUG:
		// cout << "foo ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^" << endl;
		// print_GPU<<<1,1>>>(d_cg_foo);
		// cudaDeviceSynchronize();

		// cout << "cg.cu : d_cg_res = ";
		// print_GPU<<<1,1>>>(d_cg_res); // DEBUG:
		// cudaDeviceSynchronize();

		// cout << "cg.cu : d_cg_m_minRes = ";
		// print_GPU<<<1,1>>>(d_cg_m_minRes); // DEBUG:
		// cudaDeviceSynchronize();

		// if( *res < (*m_minRed)*(*res0) )
		// 	*foo = false;

		// cout << "cg.cu : d_cg_m_minRed = ";
		// print_GPU<<<1,1>>>(d_cg_m_minRed); // DEBUG:
		// cudaDeviceSynchronize();

		// cout << "cg.cu : d_cg_res0 = ";
		// print_GPU<<<1,1>>>(d_cg_res0); // DEBUG:
		// cudaDeviceSynchronize();

		// if( *step > *m_maxIter)
		// 	*foo = false;

		// cout << "cg.cu : d_cg_step = ";
		// print_GPU<<<1,1>>>(d_cg_step); // DEBUG:
		// cudaDeviceSynchronize();

		// cout << "cg.cu : d_cg_m_maxIter = ";
		// print_GPU<<<1,1>>>(d_cg_m_maxIter); // DEBUG:
		// cudaDeviceSynchronize();


	}

	// cout << "CG:: check convergence" << endl;
	// check convergence
	// __global__ void checkConvergence(bool* foo, size_t* step, size_t* m_maxIter){
		checkConvergence<<<1,1>>>(d_cg_foo, d_cg_step, d_cg_m_maxIter);
		CUDA_CALL( cudaMemcpy( &foo, d_cg_foo, sizeof(bool), cudaMemcpyDeviceToHost) 	);
		// cudaDeviceSynchronize();
		
		if (foo)
		{
			if (m_bVerbose && mpi::world().rank() == 0)
				cout << "## CG solver failed. No convergence. "<< string(36, '#') << endl;
			return false;
		}

		if (m_bVerbose && mpi::world().rank() == 0)
		{
			cout << string(2, '#') << " CG Iteration converged. " << string(56, '#') << endl << endl;
		}

		// cout << "### Checking d_x in cg.cu ### " << endl;
		// cudaDeviceSynchronize();
		// printVector_GPU<<<1,A.num_rows()>>>(d_x);
		// cudaDeviceSynchronize();

		// cout << "cg.cu : end of solve() " << endl;
		// cudaDeviceSynchronize();

	return true;
}

bool CG::deallocation_GPU()
{
	
	CUDA_CALL( cudaFree( d_cg_p ) 			);
	CUDA_CALL( cudaFree( d_cg_r ) 			);
	CUDA_CALL( cudaFree( d_cg_z ) 			);
	CUDA_CALL( cudaFree( d_cg_res0 ) 		);
	CUDA_CALL( cudaFree( d_cg_res ) 		);
	CUDA_CALL( cudaFree( d_cg_lastRes ) 	);
	CUDA_CALL( cudaFree( d_cg_m_minRes ) 	);
	CUDA_CALL( cudaFree( d_cg_m_minRed ) 	);
	CUDA_CALL( cudaFree( d_cg_step ) 		);
	CUDA_CALL( cudaFree( d_cg_m_maxIter ) 	);
	CUDA_CALL( cudaFree( d_cg_foo ) 		);
	CUDA_CALL( cudaFree( d_cg_value ) 		);
	CUDA_CALL( cudaFree( d_cg_index ) 		);
	CUDA_CALL( cudaFree( d_cg_rho ) 		);
	CUDA_CALL( cudaFree( d_cg_rho_old ) 	);
	CUDA_CALL( cudaFree( d_cg_alpha ) 		);
	CUDA_CALL( cudaFree( d_cg_alpha_temp ) 	);
	cudaDeviceSynchronize();

	return true;

}


bool CG::solve(Vector<double>& x, const Vector<double>& b) const
{
	double* d_cg_x = nullptr;
	double* d_cg_b = nullptr;

	CUDA_CALL( cudaMalloc( (void**)&d_cg_x, sizeof(double)*x.size() ) 	);
	CUDA_CALL( cudaMalloc( (void**)&d_cg_b, sizeof(double)*b.size() ) 	);

	CUDA_CALL( cudaMemcpy( d_cg_x, &x[0], x.size() * sizeof(double), cudaMemcpyHostToDevice) 	);
	CUDA_CALL( cudaMemcpy( d_cg_b, &b[0], b.size() * sizeof(double), cudaMemcpyHostToDevice) 	);

	return true;

	// TODO:
	// return solve_GPU(d_cg_x, d_cg_b);
}

//TODO:
bool CG::precond_GPU(double* c, double* r)
{
	
	cout << "CG::precond_GPU() called" << endl;

	return true;
}
