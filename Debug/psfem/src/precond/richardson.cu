/*
 * richardson.cpp
 *
 * author: a.vogel@rub.de
 *
 */

#include <cstddef>
#include <cassert>
#include <iostream>

#include "richardson.h"

__global__ void Richardson_Precond_GPU(double* c, double* m_eta, double* r, std::size_t num_rows){

	int id = blockDim.x * blockIdx.x + threadIdx.x;

	if ( id < num_rows)
		c[id] = (*m_eta) * r[id];

}

Richardson::Richardson() : m_eta(1.0){

	cudaMalloc( (void**)&d_m_eta, sizeof(double) );
	
	cudaMemcpy( d_m_eta, &m_eta, sizeof(double), cudaMemcpyHostToDevice );
}


Richardson::Richardson(const double eta) : m_eta(eta){

cudaMalloc( (void**)&d_m_eta, sizeof(double) );

cudaMemcpy( d_m_eta, &m_eta, sizeof(double), cudaMemcpyHostToDevice );
}

bool Richardson::init(const ELLMatrix<double>& mat) 
{
	this->m_pA = &mat;

	// copy vector size

	m_num_rows = mat.num_rows();
	return true;
}

bool Richardson::precond(Vector<double>& c, const Vector<double>& r) const
{
	assert(c.size() == r.size() && "Size mismatch.");

	for (std::size_t i = 0; i < r.size(); ++i)
	{
        c[i] = m_eta * r[i];
	}

	c.set_storage_type(r.get_storage_mask());

	return true;
}

bool Richardson::precond_GPU(double* c, double* r)
{

	// Calculating the required CUDA grid and block dimensions
	dim3 blockDim;
	dim3 gridDim;

	calculateDimensions(m_num_rows, blockDim, gridDim);

	Richardson_Precond_GPU<<<blockDim,gridDim>>>(c, d_m_eta, r, m_num_rows);
	
	// cudaDeviceSynchronize();

	return true;
}
