/*
 * richardson.h
 *
 * author: a.vogel@rub.de
 *
 */

#ifndef RICHARDSON_H
#define RICHARDSON_H


#include "linear_iterator.h"


class Richardson
: public LinearIterator
{
	public:
		/// constructor
		Richardson();

		/// constructor with eta
		Richardson(const double eta);

		virtual bool init(const ELLMatrix<double>& mat);

		/// @copydoc LinearIterator::apply
		virtual bool precond(Vector<double>& c, const Vector<double>& r) const;
    
        /// set scale factor
        void set_factor(double eta) {m_eta = eta;}

 		virtual LinearIterator* clone() const {return new Richardson(*this);}

		 //DEBUG:
		bool precond_GPU(double* c, double* r);
   
    protected:
        /// scale factor (should be: 2./(largest eigenvalue))
        double m_eta;
		size_t m_num_rows;

		// CUDA device pointers
		double* d_m_eta;
};

#ifdef CUDA
	#include "richardson.cu"
#endif

#endif // RICHARDSON_H
