/*
 * iterative_solver.cpp
 *
 * author: a.vogel@rub.de
 *
 */

#include "solver/iterative_solver.h"
#include "algebra/vector.h"
#include "algebra/ell_matrix.h"
#include <cassert>
#include <iostream>
#include <iomanip> // setw
#include <cmath>


// for CUDA implementation:

#include <cuda.h>
#include <cuda_runtime.h>
#include <chrono>


	#define CUDA_CALL( call )                                                                                          \
	{                                                                                                                  \
	cudaError_t err = call;                                                                                          \
	if ( cudaSuccess != err)                                                                                         \
		fprintf(stderr, "CUDA error for %s in %d of %s : %s.\n", #call , __LINE__ , __FILE__ ,cudaGetErrorString(err));\
	}



__global__ void addStep(size_t* step){

	++(*step);
}

__global__ void checkIterationConditions(bool* foo, size_t* step, double* res, double* res0, double* m_minRes, double* m_minRed, size_t* m_maxIter){

	if ( *res > *m_minRes && *res > *m_minRed*(*res0) && (*step) <= *m_maxIter )
		*foo = true;

	else
		*foo = false;
	
}

__global__ void checkConvergence(bool* foo, size_t* step, size_t* m_maxIter){

	if(step > m_maxIter)
		*foo = true;
	
}

using namespace std;


////////////////////////////////////////
// IterativeSolver
////////////////////////////////////////

IterativeSolver::IterativeSolver(const ELLMatrix<double>& mat)
: m_pLinearIterator(0), m_maxIter(10), m_minRes(1e-50), m_minRed(1e-10), m_bVerbose(true)
{
	init(mat);
}


IterativeSolver::IterativeSolver()
: m_pLinearIterator(0), m_maxIter(10), m_minRes(1e-50), m_minRed(1e-10), m_bVerbose(true)
{}


void IterativeSolver::set_linear_iterator(LinearIterator& linIter)
{
	m_pLinearIterator = &linIter;
	if(this->m_pA)
		m_pLinearIterator->init(*this->m_pA);
}

bool IterativeSolver::init(const ELLMatrix<double>& A)
{
	this->m_pA = &A;

	res0 = 0;
	step = 0;
	
	// compute residuum: r = b - Ax
	std::vector<double> r(A.num_rows(), 0);
	
	// create correction
	std::vector<double> c(A.num_rows(), 0);


	const auto value = A.getValueAddress();
	const auto index = A.getIndexAddress();
	const size_t num_rows = A.num_rows();
	const size_t max_row_size = A.max_row_size();

	// Allocate memory in device

	CUDA_CALL( cudaMalloc( (void**)&d_r, sizeof(double)*r.size() ) 	);
	CUDA_CALL( cudaMalloc( (void**)&d_c, sizeof(double)*c.size() ) 	);
	CUDA_CALL( cudaMalloc( (void**)&d_res0, sizeof(double) ) 		);
	CUDA_CALL( cudaMalloc( (void**)&d_res, sizeof(double) ) 		);
	CUDA_CALL( cudaMalloc( (void**)&d_lastRes, sizeof(double) ) 	);
	CUDA_CALL( cudaMalloc( (void**)&d_m_minRes, sizeof(double) ) 	);
	CUDA_CALL( cudaMalloc( (void**)&d_m_minRed, sizeof(double) ) 	);
	CUDA_CALL( cudaMalloc( (void**)&d_step, sizeof(size_t) ) 		);
	CUDA_CALL( cudaMalloc( (void**)&d_m_maxIter, sizeof(size_t) ) 	);
	CUDA_CALL( cudaMalloc( (void**)&d_value, A.max_row_size() * A.num_rows() * sizeof(double) ) 		);
	CUDA_CALL( cudaMalloc( (void**)&d_index, A.max_row_size() * A.num_rows() * sizeof(std::size_t) ) 	);

	// Copy memory to device

	CUDA_CALL( cudaMemcpy( d_r, &r[0], r.size() * sizeof(double), cudaMemcpyHostToDevice) 	);
	CUDA_CALL( cudaMemcpy( d_c, &c[0], c.size() * sizeof(double), cudaMemcpyHostToDevice) 	);
	CUDA_CALL( cudaMemcpy( d_res0, &res0, sizeof(double), cudaMemcpyHostToDevice) 			);
	CUDA_CALL( cudaMemcpy( d_res, &res, sizeof(double), cudaMemcpyHostToDevice) 			);
	CUDA_CALL( cudaMemcpy( d_lastRes, &lastRes, sizeof(double), cudaMemcpyHostToDevice) 	);
	CUDA_CALL( cudaMemcpy( d_m_minRes, &m_minRes, sizeof(double), cudaMemcpyHostToDevice) 	);
	CUDA_CALL( cudaMemcpy( d_m_minRed, &m_minRed, sizeof(double), cudaMemcpyHostToDevice) 	);
	CUDA_CALL( cudaMemcpy( d_step, &step, sizeof(size_t), cudaMemcpyHostToDevice) 			);
	CUDA_CALL( cudaMemcpy( d_m_maxIter, &m_maxIter, sizeof(size_t), cudaMemcpyHostToDevice) );
	CUDA_CALL( cudaMemcpy( d_value, value, A.max_row_size() * A.num_rows() * sizeof(double), cudaMemcpyHostToDevice) 		);
	CUDA_CALL( cudaMemcpy( d_index, index, A.max_row_size() * A.num_rows() * sizeof(std::size_t), cudaMemcpyHostToDevice) 	);


	if(m_pLinearIterator)
		return m_pLinearIterator->init(A);
	else
		return true;
}


void IterativeSolver::set_convergence_params
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


void IterativeSolver::set_verbose(bool verbose)
{
	m_bVerbose = verbose;
}


bool IterativeSolver::solve(Vector<double>& x, const Vector<double>& b) const
{

	// check that stepping method has been set
	if (!m_pLinearIterator)
	{
		cout << "Step method has not been set via set_linear_iterator()." << endl;
		return false;
	}

	// check for matrix and get reference
	if(!this->m_pA){
		cout << "Matrix has not been set via set_matrix()." << endl;
		return false;
	}
	const ELLMatrix<double>& A = *(this->m_pA);

	

	// check sizes
	if(b.size() != x.size() || A.num_rows() != b.size() || A.num_cols() != x.size()){
		cout << "Iterative solver: Size mismatch." << endl;
		return false;
	}

	// [CUDA] ---------------------------------------------------------------------------

		// host variables
		
		bool foo = true;
		dim3 blockDim;
		dim3 gridDim;

		// Calculating the required CUDA grid and block dimensions
		calculateDimensions(A.num_rows(), blockDim, gridDim);
		
		// device memory allocation

		CUDA_CALL( cudaMalloc( (void**)&d_b, sizeof(double)*b.size() ) 	);
		CUDA_CALL( cudaMalloc( (void**)&d_x, sizeof(double)*x.size() ) 	);
		CUDA_CALL( cudaMalloc( (void**)&d_foo, sizeof(bool) ) 			);
		
		// host-device memory copy
		CUDA_CALL( cudaMemcpy( d_x, &x[0], x.size() * sizeof(double), cudaMemcpyHostToDevice) 	);
		CUDA_CALL( cudaMemcpy( d_b, &b[0], b.size() * sizeof(double), cudaMemcpyHostToDevice) 	);
		CUDA_CALL( cudaMemcpy( d_foo, &foo, sizeof(bool), cudaMemcpyHostToDevice) 				);


		// ComputeResiduum(r, b, A, x);
		ComputeResiduum_GPU<<<gridDim,blockDim>>>(A.num_rows(), A.max_row_size(), d_value, d_index, d_x, d_r, d_b);


		// DEBUG:
		// cout << "d_r = " << endl;
		// printVector_GPU<<<gridDim,blockDim>>>( d_r, A.num_rows() );
		// cudaDeviceSynchronize();


		// init convergence criteria
		// double res0 = r.norm();
		// norm_GPU<<<gridDim,blockDim>>>(d_res0, d_r, A.num_rows());

		//TODO:
		norm_GPU(d_res0, d_r, A.num_rows(), gridDim, blockDim);
		// norm_GPU<<<gridDim, blockDim>>>(d_res0, d_r, A.num_rows());
		// cudaDeviceSynchronize();

		// cudaEventRecord( stop, 0 );
		// cudaEventSynchronize( stop );
		// cudaEventElapsedTime ( &elapsedTime , start, stop );
		// printf("GPU-time : norm_GPU : %f milliseconds\n", elapsedTime ) ;

		// cudaDeviceSynchronize();
		// end = std::chrono::steady_clock::now();
    	// std::cout << "norm_GPU_test = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() <<std::endl;

		// cudaDeviceSynchronize();
		// begin = std::chrono::steady_clock::now();
		// cudaEventRecord( start, 0 );


		// double res = res0;
		equals_GPU<<<1,1>>>(d_res, d_res0);			
		// cudaDeviceSynchronize();

		// cudaEventRecord( stop, 0 );
		// cudaEventSynchronize( stop );
		// cudaEventElapsedTime ( &elapsedTime , start, stop );
		// printf("GPU-time : equals_GPU : %f milliseconds\n", elapsedTime ) ;

		// cudaDeviceSynchronize();
		// end = std::chrono::steady_clock::now();
    	// std::cout << "equals_GPU = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() <<std::endl;




		if (m_bVerbose && mpi::world().rank() == 0)
		{
			cout << endl;
			cout << "## Iterative solver ############################################################" << endl;
			cout << "  Iter     Residuum       Required       Rate        Reduction     Required" << endl;
		}
			
		printInitialResult_GPU<<<1,1>>>(d_res0, d_m_minRes, d_m_minRed);
		// cudaDeviceSynchronize();
		
		addStep<<<1,1>>>(d_step);



	//DEBUG:
	while(foo)
	{	
		// DEBUG:
		// cout << "i_s.cu : ################################# start of foo() at step ";
		// print_GPU<<<1,1>>>(d_step);
		// cudaDeviceSynchronize();

		// DEBUG:
		// cout << "i_s.cu : in foo, before precond" << endl;
		// cudaDeviceSynchronize();

		// cout << "d_c = " << endl;
		// printVector_GPU<<< 1, num_rows >>>( d_c );
		// cudaDeviceSynchronize();

		// DEBUG:
		// cout << "d_r = " << endl;
		// printVector_GPU<<< 1, num_rows >>>( d_r );
		// cudaDeviceSynchronize();
		
		// cudaEventRecord( start, 0 );
		// cudaDeviceSynchronize();
		// begin = std::chrono::steady_clock::now();

		// CUDA: compute correction and update defect
		if (!m_pLinearIterator->precond_GPU(d_c, d_r))
		{
			cout << "Iterative solver failed. Step method failure "
			"in step " << step << "." << endl;
			return false;
		}
		
		// cudaEventRecord( stop, 0 );
		// cudaEventSynchronize( stop );
		// cudaEventElapsedTime ( &elapsedTime , start, stop );
		// printf("GPU-time : precond_GPU : %f milliseconds\n", elapsedTime ) ;

		// cudaDeviceSynchronize();
		// end = std::chrono::steady_clock::now();
    	// std::cout << "precond_GPU = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() <<std::endl;





		// DEBUG:
		// std::cout <<"i_s.cu : after iterative solver precond" << std::endl;
		// std::cout <<"i_s.cu : c.norm()" << std::endl;
		// norm_GPU<<<gridDim,blockDim>>>(d_res, d_c, num_rows);
		// cudaDeviceSynchronize();
		// print_GPU<<<1,1>>>(d_res);
		// cudaDeviceSynchronize();

		// std::cout <<"d_c" << std::endl;
		// printVector_GPU<<<1,num_rows>>>(d_c);
		// cudaDeviceSynchronize();
		// std::cout <<"d_x" << std::endl;
		// printVector_GPU<<<1,num_rows>>>(d_x);
		// cudaDeviceSynchronize();
		// std::cout <<"d_r" << std::endl;
		// printVector_GPU<<<1,num_rows>>>(d_r);
		// cudaDeviceSynchronize();

		// cudaDeviceSynchronize();
		// begin = std::chrono::steady_clock::now();
		// cudaEventRecord( start, 0 );
		// cout << "i_s.cu : add correction to solution" << endl;
		// add correction to solution
		// x += c;
		addVector_GPU<<<gridDim,blockDim>>>( d_x, d_c, A.num_rows() );
		// cudaDeviceSynchronize();

		// cudaEventRecord( stop, 0 );
		// cudaEventSynchronize( stop );
		// cudaEventElapsedTime ( &elapsedTime , start, stop );
		// printf("GPU-time : addVector_GPU : %f milliseconds\n", elapsedTime ) ;

		// cudaDeviceSynchronize();
		// end = std::chrono::steady_clock::now();
    	// std::cout << "addVector_GPU = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() <<std::endl;


		// cout << "d_x = " << endl;
		// printVector_GPU<<< 1, num_rows >>>( d_x );
		// cudaDeviceSynchronize();
		// cout << "d_c = " << endl;
		// printVector_GPU<<< 1, num_rows >>>( d_c );
		// cudaDeviceSynchronize();
		// cout << "d_value = " << endl;
		// printVector_GPU<<< 1, 125 >>>( d_value );
		// cudaDeviceSynchronize();
		// cout << "d_index = " << endl;
		// printVector_GPU<<< 1, 125 >>>( d_index );
		// cudaDeviceSynchronize();

		// cout << "i_s.cu : UpdateResiduum_GPU" << endl;	
		// cudaDeviceSynchronize();

		// cudaEventRecord( start, 0 );

		// cudaDeviceSynchronize();
		// begin = std::chrono::steady_clock::now();

		// update residuum r = r - A*c
		UpdateResiduum_GPU<<<gridDim,blockDim>>>( A.num_rows(), A.max_row_size(), d_value, d_index, d_c, d_r );
		// cudaDeviceSynchronize();
		

		// cudaEventRecord( stop, 0 );
		// cudaEventSynchronize( stop );
		// cudaEventElapsedTime ( &elapsedTime , start, stop );
		// printf("GPU-time : UpdateResiduum_GPU : %f milliseconds\n", elapsedTime ) ;

		// cudaDeviceSynchronize();
		// end = std::chrono::steady_clock::now();
    	// std::cout << "UpdateResiduum_GPU = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() <<std::endl;



		// cout << "d_r = " << endl;
		// printVector_GPU<<< 1, num_rows >>>( d_r );
		// cudaDeviceSynchronize();

		// cudaEventRecord( start, 0 );
		// cudaDeviceSynchronize();
		// begin = std::chrono::steady_clock::now();

		// remember last residuum norm
		// lastRes = res;
		equals_GPU<<<1,1>>>(d_lastRes, d_res);
		// cudaDeviceSynchronize();
		
		// cudaEventRecord( stop, 0 );
		// cudaEventSynchronize( stop );
		// cudaEventElapsedTime ( &elapsedTime , start, stop );
		// printf("GPU-time : equals_GPU : %f milliseconds\n", elapsedTime ) ;

		// cudaDeviceSynchronize();
		// end = std::chrono::steady_clock::now();
    	// std::cout << "equals_GPU = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() <<std::endl;



		// cudaEventRecord( start, 0 );
		// cudaDeviceSynchronize();
		// begin = std::chrono::steady_clock::now();

		// compute new residuum norm
		// 	res = r.norm();
		// norm_GPU<<<gridDim,blockDim>>>(d_res, d_r, A.num_rows());
		//TODO:
		norm_GPU(d_res, d_r, A.num_rows(), gridDim, blockDim);
		// cudaDeviceSynchronize();

		// cudaEventRecord( stop, 0 );
		// cudaEventSynchronize( stop );
		// cudaEventElapsedTime ( &elapsedTime , start, stop );
		// printf("GPU-time : norm_GPU : %f milliseconds\n", elapsedTime ) ;

		// cudaDeviceSynchronize();
		// end = std::chrono::steady_clock::now();
    	// std::cout << "norm_GPU_test = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() <<std::endl;

				
		//DEBUG:
		printResult_GPU<<<1,1>>>(d_step, d_res, d_m_minRes, d_lastRes, d_res0, d_m_minRed);
		
		addStep<<<1,1>>>(d_step);

		// cudaDeviceSynchronize();
		// begin = std::chrono::steady_clock::now();
		// cudaEventRecord( start, 0 );

		checkIterationConditions<<<1,1>>>(d_foo, d_step, d_res, d_res0, d_m_minRes, d_m_minRed, d_m_maxIter);
		// cudaEventRecord( stop, 0 );
		// cudaEventSynchronize( stop );
		// cudaEventElapsedTime ( &elapsedTime , start, stop );
		// printf("GPU-time : checkIterationConditions : %f milliseconds\n", elapsedTime ) ;		

		// cudaDeviceSynchronize();
		// end = std::chrono::steady_clock::now();
    	// std::cout << "checkIterationConditions = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() <<std::endl;



		// cudaDeviceSynchronize();
		// end = std::chrono::steady_clock::now();
		// cudaEventRecord( start, 0 );

		// cout << "i_s.cu : end of foo" << endl;
		// cout << "d_foo = ";
		// print_GPU<<<1,1>>>(d_foo);
		CUDA_CALL( cudaMemcpy( &foo, d_foo, sizeof(bool), cudaMemcpyDeviceToHost) 	);
		// cudaDeviceSynchronize();

		// cudaEventRecord( stop, 0 );
		// cudaEventSynchronize( stop );
		// cudaEventElapsedTime ( &elapsedTime , start, stop );
		// cudaEventDestroy ( start );
		// cudaEventDestroy ( stop );
		// printf("GPU-time : cudaMemcpy(foo) : %f milliseconds\n", elapsedTime ) ;

		// cudaDeviceSynchronize();
		// end = std::chrono::steady_clock::now();
    	// std::cout << "cudaMemcpy(foo) = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() <<std::endl;

	}
	
	
	// check convergence
	// __global__ void checkConvergence(bool* foo, size_t* step, size_t* m_maxIter){
		checkConvergence<<<1,1>>>(d_foo, d_step, d_m_maxIter);
		CUDA_CALL( cudaMemcpy( &foo, d_foo, sizeof(bool), cudaMemcpyDeviceToHost) 	);
		
		if (foo)
		{
			if (m_bVerbose && mpi::world().rank() == 0)
				cout << "## Iterative solver failed. No convergence. "<< string(36, '#') << endl;
			return false;

		}

		if (m_bVerbose && mpi::world().rank() == 0)
		{
			cout << string(2, '#') << " Iteration converged. " << string(56, '#') << endl << endl;
		}
	

	
	// cout << "checking that Ax = b" << endl;

	// DEBUG: checking Ax = b
	// d_c = A*d_x
	
	// Apply_GPU<<<gridDim,blockDim>>>( A.num_rows(), A.max_row_size(), d_value, d_index, d_x, d_c );

	// norm_GPU_test(d_res, d_c, A.num_rows(), gridDim, blockDim);
	
	// // DEBUG: check norm of d_c
	// cout << "(A*d_x).norm() = " << endl;
	// print_GPU<<<1,1>>>( d_res );
	// cudaDeviceSynchronize();

	// norm_GPU<<<gridDim,blockDim>>>(d_res, d_b, A.num_rows());
	
	// // DEBUG: check norm of d_b
	// cout << "d_b.norm() = " << endl;
	// print_GPU<<<1,1>>>( d_res );
	// cudaDeviceSynchronize();
	
	// norm_GPU<<<gridDim,blockDim>>>(d_res, d_x, A.num_rows());
	// // DEBUG: check norm of d_b
	// cout << "d_x.norm() = " << endl;
	// print_GPU<<<1,1>>>( d_res );
	// cudaDeviceSynchronize();

	// Copying d_c to host

	CUDA_CALL( cudaMemcpy( &x[0], d_x, sizeof(double) * x.size(), cudaMemcpyDeviceToHost) 	);
	// cudaDeviceSynchronize();

	return true;
}

bool IterativeSolver::deallocation_GPU()
{
	
		CUDA_CALL( cudaFree(d_r) 		);
		CUDA_CALL( cudaFree(d_x) 		);
		CUDA_CALL( cudaFree(d_b) 		);
		CUDA_CALL( cudaFree(d_c)		);
		CUDA_CALL( cudaFree(d_res0) 	);
		CUDA_CALL( cudaFree(d_m_minRes) );
		CUDA_CALL( cudaFree(d_m_minRed) );
		CUDA_CALL( cudaFree(d_step) 	);
		CUDA_CALL( cudaFree(d_m_maxIter));
		CUDA_CALL( cudaFree(d_foo) 		);
		CUDA_CALL( cudaFree(d_res) 		);
		CUDA_CALL( cudaFree(d_lastRes) 	);
		CUDA_CALL( cudaFree(d_value) 	);
		CUDA_CALL( cudaFree(d_index) 	);

	return true;
}

// TODO:
bool IterativeSolver::solve_GPU(double* d_x, double* d_b)
{
	// // check that stepping method has been set
	// if (!m_pLinearIterator)
	// {
	// 	cout << "Step method has not been set via set_linear_iterator()." << endl;
	// 	return false;
	// }

	// // check for matrix and get reference
	// if(!this->m_pA){
	// 	cout << "Matrix has not been set via set_matrix()." << endl;
	// 	return false;
	// }
	// const ELLMatrix<double>& A = *(this->m_pA);

	

	// // check sizes
	// if(b.size() != x.size() || A.num_rows() != b.size() || A.num_cols() != x.size()){
	// 	cout << "Iterative solver: Size mismatch." << endl;
	// 	return false;
	// }

	// // use the following storage types:
	// // b: additive
	// // x: consistent
	// x.change_storage_type(PST_CONSISTENT);
	// b.change_storage_type(PST_ADDITIVE);
	
	// // compute residuum: r = b - Ax
	// Vector<double> r(b.size(), 0.0, b.layouts());
	
	// // create correction
	// Vector<double> c(x.size(), 0.0, x.layouts());



	// // [CUDA] ---------------------------------------------------------------------------

	// 	// host variables
				
	// 	double res0 = 0;
	// 	double res;
	// 	double lastRes;
	// 	size_t step = 0;
	// 	bool foo = true;
					
	// 	// A-matrix variables
		
	// 	const auto value = A.getValueAddress();
	// 	const auto index = A.getIndexAddress();
	// 	const size_t num_rows = A.num_rows();
	// 	const size_t max_row_size = A.max_row_size();

	// 	// Pointers for device

	// 	double* d_r = NULL;
	// 	double* d_x = NULL;
	// 	double* d_b = NULL;
	// 	double* d_c = NULL;
	// 	double* d_res0 = NULL;
	// 	double* d_res = NULL;
	// 	double* d_lastRes = NULL;
	// 	size_t* d_step = NULL;
	// 	size_t* d_m_maxIter = NULL;
	// 	double* d_m_minRes = NULL;
	// 	double* d_m_minRed = NULL;
	// 	bool* d_foo = NULL;
		
	// 	// From A matrix:
	// 	double* d_value = NULL;
	// 	std::size_t* d_index = NULL;
			
	// 	// Allocate memory in device

	// 		CUDA_CALL( cudaMalloc( (void**)&d_r, sizeof(double)*r.size() ) 	);
	// 		CUDA_CALL( cudaMalloc( (void**)&d_x, sizeof(double)*x.size() ) 	);
	// 		CUDA_CALL( cudaMalloc( (void**)&d_b, sizeof(double)*b.size() ) 	);
	// 		CUDA_CALL( cudaMalloc( (void**)&d_c, sizeof(double)*c.size() ) 	);
	// 		CUDA_CALL( cudaMalloc( (void**)&d_res0, sizeof(double) ) 		);
	// 		CUDA_CALL( cudaMalloc( (void**)&d_res, sizeof(double) ) 		);
	// 		CUDA_CALL( cudaMalloc( (void**)&d_lastRes, sizeof(double) ) 	);
	// 		CUDA_CALL( cudaMalloc( (void**)&d_m_minRes, sizeof(double) ) 	);
	// 		CUDA_CALL( cudaMalloc( (void**)&d_m_minRed, sizeof(double) ) 	);
	// 		CUDA_CALL( cudaMalloc( (void**)&d_step, sizeof(size_t) ) 		);
	// 		CUDA_CALL( cudaMalloc( (void**)&d_m_maxIter, sizeof(size_t) ) 	);
	// 		CUDA_CALL( cudaMalloc( (void**)&d_foo, sizeof(bool) ) 			);
	// 		CUDA_CALL( cudaMalloc( (void**)&d_value, A.max_row_size() * A.num_rows() * sizeof(double) ) 		);
	// 		CUDA_CALL( cudaMalloc( (void**)&d_index, A.max_row_size() * A.num_rows() * sizeof(std::size_t) ) 	);
			
	// 	// Copy memory to device

	// 		CUDA_CALL( cudaMemcpy( d_r, &r[0], r.size() * sizeof(double), cudaMemcpyHostToDevice) 	);
	// 		CUDA_CALL( cudaMemcpy( d_x, &x[0], x.size() * sizeof(double), cudaMemcpyHostToDevice) 	);
	// 		CUDA_CALL( cudaMemcpy( d_b, &b[0], b.size() * sizeof(double), cudaMemcpyHostToDevice) 	);
	// 		CUDA_CALL( cudaMemcpy( d_c, &c[0], c.size() * sizeof(double), cudaMemcpyHostToDevice) 	);
	// 		CUDA_CALL( cudaMemcpy( d_res0, &res0, sizeof(double), cudaMemcpyHostToDevice) 			);
	// 		CUDA_CALL( cudaMemcpy( d_res, &res, sizeof(double), cudaMemcpyHostToDevice) 			);
	// 		CUDA_CALL( cudaMemcpy( d_lastRes, &lastRes, sizeof(double), cudaMemcpyHostToDevice) 	);
	// 		CUDA_CALL( cudaMemcpy( d_m_minRes, &m_minRes, sizeof(double), cudaMemcpyHostToDevice) 	);
	// 		CUDA_CALL( cudaMemcpy( d_m_minRed, &m_minRed, sizeof(double), cudaMemcpyHostToDevice) 	);
	// 		CUDA_CALL( cudaMemcpy( d_step, &step, sizeof(size_t), cudaMemcpyHostToDevice) 			);
	// 		CUDA_CALL( cudaMemcpy( d_m_maxIter, &m_maxIter, sizeof(size_t), cudaMemcpyHostToDevice) );
	// 		CUDA_CALL( cudaMemcpy( d_foo, &foo, sizeof(bool), cudaMemcpyHostToDevice) 				);
	// 		CUDA_CALL( cudaMemcpy( d_value, value, A.max_row_size() * A.num_rows() * sizeof(double), cudaMemcpyHostToDevice) 		);
	// 		CUDA_CALL( cudaMemcpy( d_index, index, A.max_row_size() * A.num_rows() * sizeof(std::size_t), cudaMemcpyHostToDevice) 	);


	// 		cout << "iterative_solver.cu : size of A = " << A.max_row_size() * A.num_rows() << endl;
	// 		cout << "r size = " << r.size() << endl;

	// 		// ComputeResiduum(r, b, A, x);
	// 		ComputeResiduum_GPU<<<1,num_rows>>>(num_rows, max_row_size, d_value, d_index, d_x, d_r, d_b);

	// 		// init convergence criteria
	// 		// double res0 = r.norm();
	// 		norm_GPU<<<1,num_rows>>>(d_res0, d_r);

	// 		// double res = res0;
	// 		vectorEquals_GPU<<<1,1>>>(d_res, d_res0);
						
	// 		cudaDeviceSynchronize();
			
	// 		if (m_bVerbose && mpi::world().rank() == 0)
	// 		{
	// 			cout << endl;
	// 			cout << "## Iterative solver ############################################################" << endl;
	// 			cout << "  Iter     Residuum       Required       Rate        Reduction     Required" << endl;
	// 		}
			
	// 		printInitialResult_GPU<<<1,1>>>(d_res0, d_m_minRes, d_m_minRed);
	// 		cudaDeviceSynchronize();
			
			
	// 		addStep<<<1,1>>>(d_step);

	// while(foo)
	// {
	// 	// CUDA: compute correction and update defect
	// 	if (!m_pLinearIterator->precond_GPU(d_c, d_r))
	// 	{
	// 		cout << "Iterative solver failed. Step method failure "
	// 		"in step " << step << "." << endl;
	// 		return false;
	// 	}
		
		
	// 	// add correction to solution
	// 	// x += c;
	// 	addCorrection_GPU<<<1,num_rows>>>(d_x,d_c);
		
	// 	cudaDeviceSynchronize();
		
		
	// 	// update residuum r = r - A*c
	// 	UpdateResiduum_GPU<<<1,A.num_rows()>>>( num_rows, max_row_size, d_value, d_index, d_c, d_r );
	// 	cudaDeviceSynchronize();
		
		
	// 	// remember last residuum norm
	// 	// lastRes = res;
	// 	vectorEquals_GPU<<<1,1>>>(d_lastRes, d_res);
		
		
	// 	// compute new residuum norm
	// 	// 	res = r.norm();
		
	// 	norm_GPU<<<1,num_rows>>>(d_res, d_r);
		
	// 	printResult_GPU<<<1,1>>>(d_step, d_res, d_m_minRes, d_lastRes, d_res0, d_m_minRed);
		
	// 	addStep<<<1,1>>>(d_step);
	// 	checkIterationConditions<<<1,1>>>(d_foo, d_step, d_res, d_res0, d_m_minRes, d_m_minRed, d_m_maxIter);
		
	// 	CUDA_CALL( cudaMemcpy( &foo, d_foo, sizeof(bool), cudaMemcpyDeviceToHost) 	);
		
	// }
	
	
	// // check convergence
	// // __global__ void checkConvergence(bool* foo, size_t* step, size_t* m_maxIter){
	// 	checkConvergence<<<1,1>>>(d_foo, d_step, d_m_maxIter);
	// 	CUDA_CALL( cudaMemcpy( &foo, d_foo, sizeof(bool), cudaMemcpyDeviceToHost) 	);
		
	// 	if (foo)
	// 	{
	// 		if (m_bVerbose && mpi::world().rank() == 0)
	// 			cout << "## Iterative solver failed. No convergence. "<< string(36, '#') << endl;
	// 		return false;

	// 	}

	// 	if (m_bVerbose && mpi::world().rank() == 0)
	// 	{
	// 		cout << string(2, '#') << " Iteration converged. " << string(56, '#') << endl << endl;
	// 	}
	





	// //////////////////////////////////////////////
	// // Device memory deallocation
	// //////////////////////////////////////////////

	// 	CUDA_CALL( cudaFree(d_r) 		);
	// 	CUDA_CALL( cudaFree(d_x) 		);
	// 	CUDA_CALL( cudaFree(d_b) 		);
	// 	CUDA_CALL( cudaFree(d_c)		);
	// 	CUDA_CALL( cudaFree(d_res0) 	);
	// 	CUDA_CALL( cudaFree(d_m_minRes) );
	// 	CUDA_CALL( cudaFree(d_m_minRed) );
	// 	CUDA_CALL( cudaFree(d_step) 	);
	// 	CUDA_CALL( cudaFree(d_m_maxIter));
	// 	CUDA_CALL( cudaFree(d_foo) 		);
	// 	CUDA_CALL( cudaFree(d_res) 		);
	// 	CUDA_CALL( cudaFree(d_lastRes) 	);
	// 	CUDA_CALL( cudaFree(d_value) 	);
	// 	CUDA_CALL( cudaFree(d_index) 	);

	cout << "iterative solver :: solve_GPU()" << endl;
	return true;
}





//NOTE: dummy function
bool IterativeSolver::precond_GPU(double* c, double* r)
{
	
	return true;
}
