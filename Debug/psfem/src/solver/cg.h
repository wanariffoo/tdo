/*
 * cg.h
 *
 * author: a.vogel@rub.de
 *
 */

#ifndef CG_H
#define CG_H


#include "algebra/vector.h"
#include "algebra/ell_matrix.h"
#include "precond/linear_iterator.h"
#include "solver/linear_solver.h"
#include <cstddef>


/** @brief The conjugate gradient solver
 * This class implements the conjugate gradient method.
 *
 * It can be preconditioned with an iterator B that is an approximation of A 
 * and typically easy to invert. The Preconditioner (if any) is specified via the 
 * LinearIterator interface. Examples for LinearIteratora are Jacobi and GaussSeidel.
 */
class CG
 : public LinearSolver
{
	public:
		/// constructor
		CG();

		/// constructor with matrix
		CG(const ELLMatrix<double>& mat);

		/// set corrector method
		void set_linear_iterator(LinearIterator& linIter);

		/** @brief Set convergence control parameters
		 *
		 * @param maxIter maximal number of iterations
		 * @param minRes  absolute residuum norm value to be reached
		 * @param minRed  residuum norm reduction factor to be reached
		 */
		void set_convergence_params(std::size_t nIter, double minDef, double minRed);

		/// Set output verbosity
		virtual void set_verbose(bool verbose);

		/// set matrix
		virtual bool init(const ELLMatrix<double>& mat);

		/** @brief Solves the linear system using the conjugate gradient method
		 * Solve Ax = b for x.
		 *
		 * @param x  solution
		 * @param b  right-hand side
		 * @return   false on failure; true otherwise
		 */
		// virtual bool solve(double* d_x, double* d_b) const;
		bool solve_GPU(double* d_x, double* d_b);
		virtual bool solve(Vector<double>& x, const Vector<double>& b) const;

		virtual LinearIterator* clone() const {return new CG(*this);}

		//DEBUG:
		bool precond_GPU(double* c, double* r);
		bool deallocation_GPU();

	private:
		LinearIterator* m_pLinearIterator;

		std::size_t m_maxIter;
		double m_minRes;
		double m_minRed;

		bool m_bVerbose;

		/// [CUDA] ##################################

		double res0 = 0;
		double res;
		double lastRes;
		size_t step = 0;
		bool foo = true;
		double rho, rho_old;

		dim3 blockDim;
		dim3 gridDim;


		// const auto value = A.getValueAddress();
		// const auto index = A.getIndexAddress();
		// const size_t num_rows = A.num_rows();
		// const size_t max_row_size = A.max_row_size();

		// device pointers

		double* d_cg_p = NULL;
		double* d_cg_r = NULL;
		double* d_cg_z = NULL;
		double* d_cg_res0 = NULL;
		double* d_cg_res = NULL;
		double* d_cg_lastRes = NULL;
		size_t* d_cg_step = NULL;
		size_t* d_cg_m_maxIter = NULL;
		double* d_cg_m_minRes = NULL;
		double* d_cg_m_minRed = NULL;
		bool* d_cg_foo = NULL;
		double* d_cg_rho = NULL;
		double* d_cg_rho_old = NULL;
		double* d_cg_alpha = NULL;
		double* d_cg_alpha_temp = NULL;

		// for the stiffness matrix
		double* d_cg_value = NULL;
		size_t* d_cg_index = NULL;

};

#ifdef CUDA
	#include "cg.cu"
#endif

#endif // CG_H
