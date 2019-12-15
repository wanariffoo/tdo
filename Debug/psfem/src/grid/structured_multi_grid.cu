/*
 * structured_multi_grid.cpp
 *
 * author: a.vogel@rub.de
 *
 */

#include "structured_multi_grid.h"

template <std::size_t dim>
StructuredMultiGrid<dim>::StructuredMultiGrid
(
	const CoordVector<dim>& vLowerBounds,
	const CoordVector<dim>& vUpperBounds,
	const CoordVector<dim, std::size_t>& vNumElemPerDim,
	const std::size_t numGridLevels,
	mpi::Comm comm
) : m_vLowerBounds(vLowerBounds), m_vUpperBounds(vUpperBounds), m_vNumElemPerDimBaseLvl(vNumElemPerDim)
{
	// one grid level is required
	if(numGridLevels == 0){
		throw std::invalid_argument("Number of grid levels in the multi grid hierarchy must be at least 1.");
	}

	// create coarsest mesh
	m_vSGrid.push_back( StructuredGrid<dim>(m_vLowerBounds, m_vUpperBounds, m_vNumElemPerDimBaseLvl, comm) );

	// refine for more grids
	for(std::size_t l = 1; l < numGridLevels; ++l)
		refine();
}


template <std::size_t dim>
std::size_t StructuredMultiGrid<dim>::num_levels() const
{
	return m_vSGrid.size();
}

template <std::size_t dim>
const StructuredGrid<dim>& StructuredMultiGrid<dim>::grid(std::size_t lev) const
{
	return m_vSGrid.at(lev);
}

template <std::size_t dim>
const CoordVector<dim>& StructuredMultiGrid<dim>::lower_bnds() const
{
	return m_vLowerBounds;
}


template <std::size_t dim>
const CoordVector<dim>& StructuredMultiGrid<dim>::upper_bnds() const
{
	return m_vUpperBounds;
}


template <std::size_t dim>
void StructuredMultiGrid<dim>::refine()
{
	// get current mesh sizes of finest grid
	CoordVector<dim, std::size_t> vNumElemPerDim = m_vNumElemPerDimBaseLvl;

	// compute refined mesh sizes on next level
	for(std::size_t lvl = 0; lvl < num_levels(); ++lvl){
		for(std::size_t d = 0; d < dim; ++d){
			vNumElemPerDim[d] = vNumElemPerDim[d] * 2;
		}
	}

	// add new grid level
	m_vSGrid.push_back( StructuredGrid<dim>(m_vLowerBounds, m_vUpperBounds, vNumElemPerDim, m_vSGrid[0].layouts()->comm()) );
}



// explicit template declarations
template class StructuredMultiGrid<1>;
template class StructuredMultiGrid<2>;
template class StructuredMultiGrid<3>;
