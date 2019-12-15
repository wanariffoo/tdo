/*
 * ilu.h
 *
 * author: a.vogel@rub.de
 */

#ifndef ILU_H
#define ILU_H

#include "linear_iterator.h"


class ILU
 : public LinearIterator
{
	public:
		/// constructor
		ILU();

		/// set matrix
		virtual bool init(const ELLMatrix<double>& mat);

		/// apply incomplete LU factorization, i.e. solve (LU)^{-1} c = r
		bool precond(Vector<double>& c, const Vector<double>& r) const;

		virtual LinearIterator* clone() const {return new ILU(*this);}

		//DEBUG:
		bool precond_GPU(double* c, double* r);

	private:
		ELLMatrix<double> m_LU;
};

#endif // ILU_H
