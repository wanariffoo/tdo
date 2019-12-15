/*
 * jacobi.cpp
 *
 * author: a.vogel@rub.de
 *
 */

#include "jacobi.h"
#include "algebra/ell_matrix.h"

#include <cstddef>
#include <cassert>
#include <cmath>
#include <iostream>
// #include "../util/helper_cuda.h"

using namespace std;


__global__ void Jacobi_Precond_GPU(double* c, double* m_diag, double* r, size_t num_rows){

	int id = blockDim.x * blockIdx.x + threadIdx.x;

	if ( id < num_rows )
		c[id] = m_diag[id] * r[id];


}



Jacobi::Jacobi() : m_damp(1.0)
{}

Jacobi::Jacobi(double damp) : m_damp(damp)
{}

bool Jacobi::init(const ELLMatrix<double>& mat) 
{
	this->m_pA = &mat;

	m_diag.resize(mat.num_rows());
	m_diag.set_layouts(mat.layouts());
	m_diag.set_storage_type(mat.get_storage_mask());

		// copy diagonal into vector
		for (std::size_t i = 0; i < m_diag.size(); ++i)
			m_diag[i] = mat(i,i);

		// make diag consistent
		if (!m_diag.change_storage_type(PST_CONSISTENT)){
			std::cout << "Failed to convert matrix into diagonal consistent storage." << std::endl;
			return false;
		}

		// copy diagonal into vector
		for (std::size_t i = 0; i < m_diag.size(); ++i)
			m_diag[i] = m_damp / m_diag[i];
	
			
		// allocate and copy memory to device
		cudaMalloc( (void**)&d_m_diag, sizeof(double)*m_diag.size() );
		
		cudaMemcpy( d_m_diag, &m_diag[0], sizeof(double)*m_diag.size(), cudaMemcpyHostToDevice );
		


	//TODO:
	// if ( cudaSuccess != err){return false;}		

	return true;
}

bool Jacobi::precond(Vector<double>& c, const Vector<double>& r) const
{
	if(this->m_pA == 0){
		cout << "No matrix set to Jacobi linear iterator.";
		return false;
	}
	
	if(r.size() != m_diag.size() || c.size() != m_diag.size()){
		cout << "Jacobi: Square matrix columns must match Vector size. (Not init?)" << endl;
		return false;
	}
	
	// some small number
	// const double eps = 1e-50;
	// \todo: add check for size of diag elements
	
	// work directly with consistent diag vector
	const std::size_t sz = r.size();

			for (std::size_t i = 0; i < sz; ++i){
				c[i] = m_diag[i] * r[i];
			}			

	c.set_storage_type(r.get_storage_mask());

	return true;
}

bool Jacobi::precond_GPU(double* c, double* r)
{
	if(this->m_pA == 0){
		cout << "No matrix set to Jacobi linear iterator.";
		return false;
	}

	// Calculating the required CUDA grid and block dimensions
	dim3 blockDim;
	dim3 gridDim;

	calculateDimensions(m_diag.size(), blockDim, gridDim);
	
	Jacobi_Precond_GPU<<<gridDim,blockDim>>>(c, d_m_diag, r, m_diag.size());
	// cudaDeviceSynchronize();

	
	return true;
}

