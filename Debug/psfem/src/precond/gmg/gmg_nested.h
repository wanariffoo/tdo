/*
 * gmg.h
 *
 * author: a.vogel@rub.de
 *
 */

#ifndef GMG_NESTED_H
#define GMG_NESTED_H


#include "gmg.h"
#include "prolongation.h"
#include "precond/linear_iterator.h"
#include "solver/linear_solver.h"
#include "grid/structured_multi_grid.h"
#include "disc/assemble_interface.h"
#include "algebra/ell_matrix.h"
#include "algebra/vector.h"

template<std::size_t dim>
class NestedGMG : public GMG<dim>
{
	protected:
	using GMG<dim>::m_vSmoother;
	using GMG<dim>::m_multiGrid;
	using GMG<dim>::m_baseLvl;
	using GMG<dim>::m_vStiffMat;
	using GMG<dim>::m_pBaseSolver;
	using GMG<dim>::m_pSmoother;
	using GMG<dim>::m_pDisc;
	using GMG<dim>::m_vProlongMat;
	using GMG<dim>::m_pProl;
	using GMG<dim>::precond_add_update;
	using GMG<dim>::m_gamma;

	public:
		/// constructor
		NestedGMG(	StructuredMultiGrid<dim>& multiGrid,
					IAssemble<dim>& disc,
					IProlongation<dim>& prol,
					LinearIterator& smoother,
					LinearSolver& baseSolver);

		/// set number of GMG iterations on each level
		void set_level_iterations(std::size_t n);

		/// @copydoc LinearIterator::init
		virtual bool init(const ELLMatrix<double>& mat);

		/// @copydoc LinearIterator::apply
		bool solve(Vector<double>& x) const;

		virtual LinearIterator* clone() const {return new NestedGMG(*this);}

		//DEBUG:
		bool precond_GPU(double* c, double* r);

	protected:
		std::size_t m_levelIter;

		std::vector<Vector<double>> m_vRhs;
};

#ifdef CUDA
	#include "gmg_nested.cu"
#endif

#endif // GMG_NESTED_H
