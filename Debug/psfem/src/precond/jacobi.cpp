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

using namespace std;

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

	//////////////////////////
	// parallel case
	//////////////////////////
	if(mat.layouts()->comm().size() > 1)
	{
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
	}
	//////////////////////////
	// serial case
	//////////////////////////
	else
	{
		// copy diagonal into vector
		for (std::size_t i = 0; i < m_diag.size(); ++i)
			m_diag[i] = m_damp / mat(i,i);

		m_diag.set_storage_type(PST_CONSISTENT | PST_ADDITIVE | PST_UNIQUE);
	}

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


//NOTE: dummy function
bool Jacobi::precond_GPU(double* c, double* r)
{
	
	return true;
}