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

	// use the following storage types:
	// b: additive
	// x: consistent
	x.change_storage_type(PST_CONSISTENT);
	b.change_storage_type(PST_ADDITIVE);

	// compute residuum: r = b - Ax
	Vector<double> r(b.size(), 0.0, b.layouts());
	ComputeResiduum(r, b, A, x);

	// create correction
	Vector<double> c(x.size(), 0.0, x.layouts());

	// init convergence criteria
	double res0 = r.norm();
	double res = res0;
	double lastRes;
	size_t step = 0;

	if (m_bVerbose && mpi::world().rank() == 0)
	{
		cout << endl;
		cout << "## Iterative solver ############################################################" << endl;
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
		// compute correction and update defect
		if (!m_pLinearIterator->precond(c, r))
		{
			cout << "Iterative solver failed. Step method failure "
					"in step " << step << "." << endl;
			return false;
		}

		// make correction consistent
		c.change_storage_type(PST_CONSISTENT);

		// add correction to solution
		x += c;

		// update residuum r = r - A*c
		UpdateResiduum(r, A, c);

		// remember last residuum norm
		lastRes = res;

		// compute new residuum norm
		res = r.norm();

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
			cout << "## Iterative solver failed. No convergence. "<< string(36, '#') << endl;
		return false;
	}

	if (m_bVerbose && mpi::world().rank() == 0)
	{
		cout << string(2, '#') << " Iteration converged. " << string(56, '#') << endl << endl;
	}

	return true;
}

//NOTE: dummy function
bool IterativeSolver::precond_GPU(double* c, double* r)
{
	
	return true;
}

