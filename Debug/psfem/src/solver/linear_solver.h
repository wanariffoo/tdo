/*
 * linear_solver.h
 *
 * author: a.vogel@rub.de
 *
 */

#ifndef LINEAR_SOLVER_H
#define LINEAR_SOLVER_H

#include "algebra/vector.h"
#include "precond/linear_iterator.h"

/** @brief Interface class for linear solver
 * Each class (LU, LinearIterator, CG, ...) implementing this interface has to
 * provide the method solve() which will compute the solution.
 */
class LinearSolver
	: public LinearIterator
{
	public:
		/// constructor
		LinearSolver() {};

		/// destructor
		virtual ~LinearSolver() {};

		/** @brief Solves the linear system
		 * Solve Ax = b for x.
		 *
		 * @param x  output: solution
		 * @param b  input: right-hand side
		 * @return   false on any failure; true otherwise
		 */
		virtual bool solve(Vector<double>& x, const Vector<double>& b) const = 0;
		virtual bool solve_GPU(double* d_x, double* d_b){return true;};

		virtual bool precond(Vector<double>& c, const Vector<double>& r) const {
			c = 0.0;
			return solve(c,r);
		}

		virtual bool deallocation_GPU() = 0;

};

#endif // LINEAR_SOLVER_H
