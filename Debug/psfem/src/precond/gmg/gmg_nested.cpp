/*
 * gmg.cpp
 *
 * author: a.vogel@rub.de
 *
 */

#include <cstddef>
#include <cassert>
#include <iostream>
#include "gmg_nested.h"
#include "util/vtk.h"

template <std::size_t dim>
NestedGMG<dim>::NestedGMG
(
		StructuredMultiGrid<dim>& multiGrid,
		IAssemble<dim>& disc,
		IProlongation<dim>& prol,
		LinearIterator& smoother,
		LinearSolver& baseSolver
)
: GMG<dim>(multiGrid, disc, prol, smoother, baseSolver), m_levelIter(1)
{
}


template <std::size_t dim>
void NestedGMG<dim>::set_level_iterations(std::size_t n)
{
	m_levelIter = n;
}


template <std::size_t dim>
bool NestedGMG<dim>::init(const ELLMatrix<double>& mat)
{
	// assemble prolongation
	m_vProlongMat.resize(m_multiGrid.num_levels() - 1);
	for(std::size_t lev = 0; lev < m_vProlongMat.size(); ++lev){
		m_pProl->assemble(m_vProlongMat[lev], m_multiGrid.grid(lev+1), m_multiGrid.grid(lev));
	}

	// assemble matrices
	m_vStiffMat.resize(m_multiGrid.num_levels());
	m_vRhs.resize(m_multiGrid.num_levels());
	Vector<double> dummyX;
	for(std::size_t lev = 0; lev < m_vStiffMat.size(); ++lev)
		m_pDisc->assemble(m_vStiffMat[lev], m_vRhs[lev], dummyX, m_multiGrid.grid(lev));

	// init base solver
	m_pBaseSolver->init(m_vStiffMat[m_baseLvl]);

	// init smoother
	m_vSmoother.resize(m_multiGrid.num_levels());
	for(std::size_t lev = 0; lev < m_vSmoother.size(); ++lev)
	{
		m_vSmoother[lev].reset(m_pSmoother->clone());
		m_vSmoother[lev]->init(m_vStiffMat[lev]);
	}

	return true;
}


template <std::size_t dim>
bool NestedGMG<dim>::solve(Vector<double>& x) const
{
    Vector<double> r = m_vRhs[m_baseLvl];

	// coarse solution vector
	Vector<double> u_c(r.size(), 0.0, r.layouts());

	// base level
    if (!precond_add_update(u_c, r, m_baseLvl, m_gamma))
	{
		std::cout << "Nested GMG failed on base level. Aborting." << std::endl;
		return false;
	}

	if(x.layouts()->comm().rank() == 0)
	   	std::cout << "Nested iteration: solved on level " << 0 << std::endl;

	// fine solution
	Vector<double> u_f;

    // iterate levels to the top
    const std::size_t nLvl = m_multiGrid.num_levels();
    for (std::size_t lvl = m_baseLvl+1; lvl < nLvl; ++lvl)
    {
    	// prolongation of coarse solution to fine level
    	m_pProl->interpolate(u_f, u_c, m_multiGrid.grid(lvl), m_multiGrid.grid(lvl-1));

		// calculate initial defect for GMG on this level
	    // r = m_vRhs[lvl] - m_vStiffMat[lvl] * u_f;
	    ComputeResiduum(r, m_vRhs[lvl], m_vStiffMat[lvl], u_f);

		// apply GMG k times
    	Vector<double> c(u_f.size(), 0.0, u_f.layouts());
    	for (std::size_t i = 0; i < m_levelIter; ++i)
    	{
    		if (!precond_add_update(c, r, lvl, m_gamma))
			{
				std::cout << "Nested GMG failed on level " << lvl << ". Aborting." << std::endl;
				return false;
			}
    	}
		u_f += c;

    	// copy fine to new coarse sol
    	u_c = u_f;

 		if(x.layouts()->comm().rank() == 0)
	   		std::cout << "Nested iteration: solved on level " << lvl << std::endl;
    }

    x = u_c;

    return true;
}

//NOTE: dummy function
template<std::size_t dim>
bool NestedGMG<dim>::precond_GPU(double* c, double* r)
{
	
	return true;
}



// explicit template declarations
template class NestedGMG<1>;
template class NestedGMG<2>;
template class NestedGMG<3>;
