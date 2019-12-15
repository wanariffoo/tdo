/*
 * linear_iterator.h
 *
 * author: a.vogel@rub.de
 *
 */

#ifndef LINEAR_ITERATOR_H
#define LINEAR_ITERATOR_H

#include "algebra/vector.h"
#include "algebra/ell_matrix.h"

/** @brief Interface class for iterative corrector methods
 * Each class (Jacobi, Gauss-Seidel etc.) implementing this interface has to
 * provide the method apply() which will perform one step of the method
 * resulting in a correction that can be used, e.g., in an IterativeSolver.
 */
class LinearIterator
{
	public:
		/// constructor
		LinearIterator() : m_pA(NULL) {};

		/// destructor
		virtual ~LinearIterator() {};

		/// set matrix
		virtual bool init(const ELLMatrix<double>& mat) {m_pA = &mat; return true;}

		/** @brief Apply linear iterator
		 * This method applies the linear iterator. Given a residuum r, 
		 * is computes a correction c = B^{-1} r
		 * using a linear iterator B (approximation of A).
		 *
		 * @param c  correction
		 * @param r  residuum
		 * @return   false on failure; true otherwise
		 */
		virtual bool precond(Vector<double>& c, const Vector<double>& r) const = 0;
		virtual bool precond_GPU(double* c, double* r) = 0;

		virtual LinearIterator* clone() const = 0;

	protected:
		/// underlying matrix
		const ELLMatrix<double>* m_pA;
};


#endif // LINEAR_ITERATOR_H
