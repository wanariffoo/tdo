/*
 * gmg.cpp
 *
 * author: a.vogel@rub.de
 *
 */

#include "gmg.h"
#include <cstddef>
#include <cassert>
#include <iostream>

using namespace std;

template <std::size_t dim>
GMG<dim>::GMG(
	StructuredMultiGrid<dim> &multiGrid,
	IAssemble<dim> &disc,
	IProlongation<dim> &prol,
	LinearIterator &smoother,
	LinearSolver &baseSolver)
	: m_multiGrid(multiGrid), m_pDisc(&disc),
	  m_pProl(&prol), m_pSmoother(&smoother), m_pBaseSolver(&baseSolver)
{
	// default number of smoothing steps
	set_num_smooth(1);
	set_cycle(1);
	set_base_level(0);
	set_rap(false);
}

template <std::size_t dim>
void GMG<dim>::set_rap(bool bRAP)
{
	m_bRAP = bRAP;
}

template <std::size_t dim>
void GMG<dim>::set_base_level(std::size_t lvl)
{

	if (lvl >= m_multiGrid.num_levels())
	{
		std::cout << "BaseLvl = " << lvl << " requested, but only " << m_multiGrid.num_levels() << " available." << std::endl;
		throw std::invalid_argument("BaseLvl: invalid argument");
	}

	m_baseLvl = lvl;
}

template <std::size_t dim>
void GMG<dim>::set_cycle(const char type)
{
	switch (type)
	{
	case 'V':
		m_gamma = 1;
		break;
	case 'W':
		m_gamma = 2;
		break;
	case 'F':
		m_gamma = -1;
		break;

	default:
		std::cout << "Cycle type '" << type << "' invalid argument" << std::endl;
		throw std::invalid_argument("Cycle type: invalid argument");
	}
}

template <std::size_t dim>
void GMG<dim>::set_cycle(int gamma)
{
	if (gamma < 1)
	{
		std::cout << "Gamma = " << gamma << " is invalid argument" << std::endl;
		throw std::invalid_argument("Cycle type: invalid argument");
	}
	m_gamma = gamma;
}

template <std::size_t dim>
bool GMG<dim>::init(const ELLMatrix<double> &mat)
{
	// assemble prolongation
	m_vProlongMat.resize(m_multiGrid.num_levels() - 1);
	for (std::size_t lev = 0; lev < m_vProlongMat.size(); ++lev)
	{
		m_pProl->assemble(m_vProlongMat[lev], m_multiGrid.grid(lev + 1), m_multiGrid.grid(lev));
	}

	// create coarse grid matrices
	m_vStiffMat.resize(m_multiGrid.num_levels());
	if (m_bRAP)
	{

		// copy finest matrix
		m_vStiffMat[m_multiGrid.num_levels() - 1] = mat;

		// use P^T A P
		for (std::size_t lev = m_vStiffMat.size() - 1; lev != 0; --lev)
		{
			MultiplyPTAP(m_vStiffMat[lev - 1], m_vStiffMat[lev], m_vProlongMat[lev - 1]);
			m_vStiffMat[lev - 1].set_storage_type(PST_ADDITIVE);
			m_vStiffMat[lev - 1].set_layouts(m_multiGrid.grid(lev - 1).layouts());
		}
	}
	else
	{

		// assemble matrices
		m_vStiffMat.resize(m_multiGrid.num_levels());
		Vector<double> dummyX;
		for (std::size_t lev = 0; lev < m_vStiffMat.size(); ++lev)
			m_pDisc->assemble(m_vStiffMat[lev], dummyX, m_multiGrid.grid(lev));
	}

	// init base solver
	m_pBaseSolver->init(m_vStiffMat[m_baseLvl]);

	// init smoother
	m_vSmoother.resize(m_multiGrid.num_levels());
	for (std::size_t lev = 0; lev < m_vSmoother.size(); ++lev)
	{
		m_vSmoother[lev].reset(m_pSmoother->clone());
		m_vSmoother[lev]->init(m_vStiffMat[lev]);
	}

	return true;
}

template <std::size_t dim>
bool GMG<dim>::precond_add_update(Vector<double> &c, Vector<double> &r, std::size_t lev, int cycle) const
{
	Vector<double> ctmp(c.size(), 0.0, c.layouts());

	// base solver on base level
	if (lev == m_baseLvl)
	{
		if (!m_pBaseSolver->solve(ctmp, r))
		{
			std::cout << "Base solver failed on level " << lev << ". Aborting." << std::endl;
			return false;
		}
		c += ctmp;
		// r -= m_vStiffMat[lev] * c;
		UpdateResiduum(r, m_vStiffMat[lev], ctmp);
		return true;
	}

	// presmooth
	for (std::size_t nu1 = 0; nu1 < m_numPreSmooth; ++nu1)
	{

		if (!m_vSmoother[lev]->precond(ctmp, r))
			return false;
		ctmp.change_storage_type(PST_CONSISTENT);

		c += ctmp;
		// r -= m_vStiffMat[lev] * ctmp;
		UpdateResiduum(r, m_vStiffMat[lev], ctmp);
	}

	// restrict defect
	Vector<double> r_coarse(m_vProlongMat[lev - 1].num_cols(), 0.0, m_multiGrid.grid(lev - 1).layouts());

	// d_coarse = m_vRestrictMat[lev-1] * d;
	ApplyTransposed(r_coarse, m_vProlongMat[lev - 1], r);

	// coarse grid solve
	Vector<double> c_coarse(r_coarse.size(), 0.0, r_coarse.layouts());

	if (cycle == _F_)
	{

		// one F-Cycle ...
		if (!precond_add_update(c_coarse, r_coarse, lev - 1, _F_))
		{
			std::cout << "gmg failed on level " << lev << ". Aborting." << std::endl;
			return false;
		}

		// ... followed by a V-Cycle
		if (!precond_add_update(c_coarse, r_coarse, lev - 1, _V_))
		{
			std::cout << "gmg failed on level " << lev << ". Aborting." << std::endl;
			return false;
		}
	}
	else
	{

		// V- and W-cycle
		for (int g = 0; g < cycle; ++g)
		{

			if (!precond_add_update(c_coarse, r_coarse, lev - 1, cycle))
			{
				std::cout << "gmg failed on level " << lev << ". Aborting." << std::endl;
				return false;
			}
		}
	}

	// prolongate coarse grid correction
	// ctmp = m_vProlongMat[lev-1] * c_coarse;
	Apply(ctmp, m_vProlongMat[lev - 1], c_coarse);
	ctmp.set_storage_type(PST_CONSISTENT);

	// add correction and update defect
	c += ctmp;
	// d -= m_vStiffMat[lev] * ctmp;
	UpdateResiduum(r, m_vStiffMat[lev], ctmp);

	// postsmooth
	for (std::size_t nu2 = 0; nu2 < m_numPostSmooth; ++nu2)
	{

		if (!m_vSmoother[lev]->precond(ctmp, r))
			return false;
		ctmp.change_storage_type(PST_CONSISTENT);

		c += ctmp;
		// r -= m_vStiffMat[lev] * ctmp;
		UpdateResiduum(r, m_vStiffMat[lev], ctmp);
	}

	return true;
}

template <std::size_t dim>
bool GMG<dim>::precond(Vector<double> &c, const Vector<double> &r) const
{
	if (c.size() != r.size())
	{
		cout << "GMG: Size mismatch." << endl;
		return false;
	}

	std::size_t topLev = m_multiGrid.num_levels() - 1;

	// reset correction
	//c.resize(d.size());
	c = 0.0;
	Vector<double> rtmp(r);

	return precond_add_update(c, rtmp, topLev, m_gamma);
}

// NOTE: dummy function
template <std::size_t dim>
bool GMG<dim>::precond_GPU(double *c, double *r)
{

	return true;
}

// explicit template declarations
template class GMG<1>;
template class GMG<2>;
template class GMG<3>;
