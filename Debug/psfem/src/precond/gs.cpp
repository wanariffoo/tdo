/*
 * gs.cpp
 *
 * author: a.vogel@rub.de
 *
 */

#include <cstddef>
#include <cassert>
#include <iostream>

#include "gs.h"
#include "algebra/ell_matrix.h"

using namespace std;



/// c = (L+D)^{-1} r
bool InvertLD(Vector<double>& c, const ELLMatrix<double>& A, const Vector<double>& r)
{
	if(r.size() != A.num_rows() || c.size() != A.num_cols() || r.size() != c.size()){
		cout << "Matrix columns must match Vector size." << endl;
		return false;
	}

	// some small number
	const double eps = 1e-50;

	for(size_t i = 0; i < A.num_rows(); ++i){

		double rhs = r[i];

		for(auto it = A.begin(i); it !=  A.end(i); ++it){
			if(it.index() < i){
				rhs -= c[it.index()] * it.value();
			}
		}
		
		const double diag = A(i,i);
		if(fabs(diag) < eps * fabs(rhs)){
			cout << "A(" << i << "," << i << ") = " << diag << endl;
			cout << "GaussSeidel: diagonal close to zero." << endl;
			return false;
		}

		c[i] = (1.0 / diag) * rhs;
	}

	return true;
}

/// c = (D+U)^{-1} r
bool InvertDU(Vector<double>& c, const ELLMatrix<double>& A, const Vector<double>& r)
{
	if(r.size() != A.num_rows() || c.size() != A.num_cols() || r.size() != c.size()){
		cout << "Matrix columns must match Vector size." << endl;
		return false;
	}

	// some small number
	const double eps = 1e-50;

	if(c.size() == 0) return true;
	size_t i = c.size()-1;
	do
	{
		double rhs = r[i];

		for(auto it = A.begin(i); it !=  A.end(i); ++it){
			if(it.index() > i){
				rhs -= c[it.index()] * it.value();
			}
		}
		
		const double diag = A(i,i);
		if(fabs(diag) < eps * fabs(rhs)){
			cout << "A(" << i << "," << i << ") = " << diag << endl;
			cout << "GaussSeidel: diagonal close to zero." << endl;
			return false;
		}

		c[i] = (1.0 / diag) * rhs;
		
	} while(i-- != 0);

	return true;
}


bool GaussSeidel::precond(Vector<double>& c, const Vector<double>& r) const
{
	if(c.layouts()->comm().size() > 1)
		throw(std::runtime_error("GaussSeidel: not parallelized"));	

	if(this->m_pA == 0){
		cout << "No matrix set to GaussSeidel linear iterator.";
		return false;
	}

	const ELLMatrix<double>& A = *(this->m_pA);

	// B = (L+D)
	return InvertLD(c, A, r);
}

bool BackwardGaussSeidel::precond(Vector<double>& c, const Vector<double>& r) const
{
	if(c.layouts()->comm().size() > 1)
		throw(std::runtime_error("BackwardGaussSeidel: not parallelized"));	

	if(this->m_pA == 0){
		cout << "No matrix set to GaussSeidel linear iterator.";
		return false;
	}

	const ELLMatrix<double>& A = *(this->m_pA);

	// B = (D+U)
	return InvertDU(c, A, r);
}

bool SymmetricGaussSeidel::precond(Vector<double>& c, const Vector<double>& r) const
{
	if(c.layouts()->comm().size() > 1)
		throw(std::runtime_error("SymmetricGaussSeidel: not parallelized"));	

	if(this->m_pA == 0){
		cout << "No matrix set to GaussSeidel linear iterator.";
		return false;
	}

	const ELLMatrix<double>& A = *(this->m_pA);

	// SGS is B = (D+U)^{-1} D (D+L)^{-1}
	bool bRet = true;

	// c1 = (D-L)^{-1} r
	bRet |= InvertLD(c,A,r);

	// c2 = D c1
	for(std::size_t i = 0; i <  c.size(); i++){
		c[i] = A(i,i) * c[i];
	}

	// c3 = (D+U)^{-1} c2
	// note: one can savely pass (c,c) here, since overwritten values in c2 are not needed anymore
	bRet |= InvertDU(c,A,c);

	return bRet;

}
