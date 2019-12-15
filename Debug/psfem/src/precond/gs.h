/*
 * gs.h
 *
 * author: a.vogel@rub.de
 *
 */

#ifndef GAUSS_SEIDEL_H
#define GAUSS_SEIDEL_H


#include "linear_iterator.h"

class GaussSeidel
: public LinearIterator
{
	public:
		/// constructor
		GaussSeidel() {};

		/// @copydoc LinearIterator::apply
		virtual bool precond(Vector<double>& c, const Vector<double>& r) const;

		virtual LinearIterator* clone() const {return new GaussSeidel(*this);}
		
		//DEBUG:
		bool precond_GPU(double* c, double* r);
};

class BackwardGaussSeidel
: public LinearIterator
{
	public:
		/// constructor
		BackwardGaussSeidel() {};

		/// @copydoc LinearIterator::apply
		virtual bool precond(Vector<double>& c, const Vector<double>& r) const;

		virtual LinearIterator* clone() const {return new BackwardGaussSeidel(*this);}

		//DEBUG:
		bool precond_GPU(double* c, double* r);
};

class SymmetricGaussSeidel
: public LinearIterator
{
	public:
		/// constructor
		SymmetricGaussSeidel() {};

		/// @copydoc LinearIterator::apply
		virtual bool precond(Vector<double>& c, const Vector<double>& r) const;

		virtual LinearIterator* clone() const {return new SymmetricGaussSeidel(*this);}

		//DEBUG:
		bool precond_GPU(double* c, double* r);
};


#endif // GAUSS_SEIDEL_H
