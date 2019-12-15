/*
 * cg.cpp
 *
 * author: a.vogel@rub.de
 *
 */

#include "cg.h"
#include "parallel/parallel.h"
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


bool CG::solve(Vector<double>& x, const Vector<double>& b) const
{
	// check for matrix and get reference
	if(!this->m_pA){
		cout << "Matrix has not been set via set_matrix()." << endl;
		return false;
	}
	const ELLMatrix<double>& A = *(this->m_pA);

	// check sizes
	if(b.size() != x.size() || A.num_rows() != b.size() || A.num_cols() != x.size()){
		cout << "CG: Size mismatch." << endl;
		return false;
	}

	// use the following storage types:
	// b: additive
	// x: consistent
	x.change_storage_type(PST_CONSISTENT);
	b.change_storage_type(PST_ADDITIVE);

	// create required vectors
	Vector<double> p(x.size(), 0.0, x.layouts()); // search direction
	Vector<double> r(x.size(), 0.0, x.layouts()); // residuum
	Vector<double> z(x.size(), 0.0, x.layouts()); // help vector

	// compute residuum: r = b-Ax
	ComputeResiduum(r, b, A, x);

	// initial residuum
	double res0, res, lastRes; 
	res0 = res = r.norm();

	cout << "cg.cpp : ComputeResiduum()" << endl;
	for (int i = 0; i < r.size(); ++i)
	cout << "r[" << i << "] = " << r[i] << endl;

	// step counter
	size_t step = 0;

	// help values
	double rho, rho_old;

	// output initial values
	if (m_bVerbose && mpi::world().rank() == 0)
	{
		cout << "## CG  ##################################################################" << endl;
		cout << "  Iter     Residuum       Required       Rate        Reduction     Required" << endl;
		cout << right << setw(5) << step << "    ";
		cout << scientific << res0 <<  "    ";
		cout << scientific << setprecision(3) << m_minRes <<   "    " << setprecision(6);
		cout << scientific << "  -----  " << "    ";
		cout << scientific << "  --------  " << "    ";
		cout << scientific << setprecision(3) << m_minRed << "    " << setprecision(6);
		cout << endl;
	}

	// iteration
	while (res > m_minRes && res > m_minRed*res0 && ++step <= m_maxIter)
	{
		// apply preconditioner
		if(m_pLinearIterator){
			if (!m_pLinearIterator->precond(z, r)){
				cout << "CG failed. Step method failure in step " << step << "." << endl;
				return false;
			}
		}
		else{
			z = r;
		}

		// force z to be consistent
		if (!z.change_storage_type(PST_CONSISTENT)){
			cout << "Failed to change storage type for z." << endl;
			return false;
		}
		
		// rho = (z,r)
		rho = r * z;

		// p = z + p * beta;
		if(step == 1) p = z;
		else {
			p *= (rho / rho_old);
			p += z;
		}

		// z = A*p
		Apply(z, A, p);

		// alpha = rho / (z,p)
		const double alpha = rho / (p * z);

		// add correction to solution
		// x = x + alpha * p
		x.axpy(alpha, p);

		// update residuum
		// r = r - alpha * t
		r.axpy(-alpha, z);

		// compute residuum
		lastRes = res;
		res = r.norm();

		// store old rho
		rho_old = rho;

		// output iteration progress
		if (m_bVerbose && mpi::world().rank() == 0)
		{
			cout << right << setw(5) << step << "    ";

			cout << scientific << res <<  "    ";
			cout << scientific << setprecision(3) << m_minRes <<   "    " << setprecision(6);
			cout << scientific << setprecision(3) << res / lastRes << "    "<< setprecision(6);
			cout << scientific << res / res0 << "    ";
			cout << scientific << setprecision(3) << m_minRed << "    " << setprecision(6);
			cout << endl;
		}
	}

	// check convergence
	if (step > m_maxIter )
	{
		if (m_bVerbose && mpi::world().rank() == 0)
			cout << "## CG solver failed. No convergence. "<< string(34, '#') << endl;
		return false;
	}

	if (m_bVerbose && mpi::world().rank() == 0)
	{
		cout << string(2, '#') << " CG Iteration converged. " << string(54, '#') << endl << endl;
	}


	return true;
}

//NOTE: dummy function
bool CG::precond_GPU(double* c, double* r)
{
	
	return true;
}

