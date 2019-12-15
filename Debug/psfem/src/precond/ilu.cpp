/*
 * ilu.h
 *
 * author: a.vogel@rub.de
 */

#include "ilu.h"
#include <cassert>
#include <cmath>
#include <cstddef>
#include <iostream>

ILU::ILU() {}

bool ILU::init(const ELLMatrix<double>& mat)
{
	if(mat.layouts()->comm().size() > 1)
		throw(std::runtime_error("ILU: not parallelized"));	

	// some small number
	const double eps = 1e-50;

	// store reference to original matrix
	this->m_pA = &mat;

	// store a copy of the original matrix (and a reference for easiear notation)
	m_LU = mat;
	ELLMatrix<double>& A = m_LU;

	// check matrix is quadratic
	if(A.num_rows() != A.num_cols())
		throw(std::runtime_error("ILU: Matrix expected to be square matrix"));

	// perform decomposition
	std::size_t sz = A.num_rows();
	for (std::size_t i = 1; i < sz; ++i)
	{
		for(auto it_k = A.begin(i); it_k !=  A.end(i); ++it_k){
			// loop only L entries of this row
			const std::size_t k = it_k.index();
			if (k >= i) continue;

			// get entry to eliminate
			double& A_ik = it_k.value();

			// only eliminate lower left entries if != 0
			if (!A_ik) continue;

			// get diag
			const double A_kk = A(k,k);

			// check for small entries
			if (fabs(A_kk) < eps * fabs(A_ik)){
				throw(std::runtime_error("ILU: factorization failed, small value"));
			}

			// devide by diagonal
			A_ik /= A_kk;

			// handle rest of the row
			for(auto it_j = A.begin(i); it_j !=  A.end(i); ++it_j){
				const std::size_t j = it_j.index();
				if (j <= k) continue;

				// get entry to modify
				double& A_ij = it_j.value();

				// subtract
				A_ij -= A_ik * A(k,j);
			}
		}
	}

	return true;
}


bool ILU::precond(Vector<double>& c, const Vector<double>& r) const
{
	// some small number
	const double eps = 1e-50;

	// check sizes
	if(r.size() != c.size() || m_LU.num_rows() != r.size() || m_LU.num_cols() != c.size())
		throw(std::runtime_error("ILU: size mismatch"));

	// invert L
	std::size_t sz = m_LU.num_rows();
	for (std::size_t i = 0; i < sz; ++i)
	{
		c[i] = r[i];

		// loop only L entries of this row
		for(auto it = m_LU.begin(i); it !=  m_LU.end(i); ++it){
			if (it.index() >= i) continue;

			c[i] -= it.value() * c[it.index()];
		}

		// diagonal (implicitly, not stored) is 1 for L
	}

	// invert U
	for (std::size_t i = sz-1; i < sz; --i)
	{
		// loop only U entries of this row
		for(auto it = m_LU.begin(i); it !=  m_LU.end(i); ++it){
			if (it.index() <= i) continue;

			c[i] -= it.value() * c[it.index()];
		}

		// check diagonal
		const double diag = m_LU(i,i);
		if (fabs(diag) < eps * fabs(c[i])){
			std::cout << "ILU: application failed at row " << i << ": "
					"A(" << i << "," << i << ") is practically zero." << std::endl;
			return false;
		}

		// devide by diagonal
		c[i] /= diag;
	}

	return true;
}


