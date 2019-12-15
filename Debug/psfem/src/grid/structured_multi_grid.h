/*
 * structured_multi_grid.h
 *
 * author: a.vogel@rub.de
 *
 */

#ifndef STRUCTURED_MULTI_GRID_H
#define STRUCTURED_MULTI_GRID_H

#include "util/coord_vector.h"
#include "grid/structured_grid.h"

#include <cstddef>
#include <vector>


/**
 * A structured multi grid hierarchy.
 * This class represents a structured multi grid that is equi-distant w.r.t.
 * all dimensions separately. It is defined by the lower and upper bounds
 * for all dimensions as well as the spatial resolution (i.e., number of elements)
 * for the coarsest grid level. In addition, the numbers of refinements are specified
 * resulting in a number of nested grid levels. To this purpose, internally the
 * grid levels are stored as a StructuredGrid instance and can be requested for each
 * grid level individually.
 */

template <std::size_t dim>
class StructuredMultiGrid
{
	public:
		/** @brief Constructor with bounds and resolution
		 * Constructs the grid hierarchy.
		 *
		 * @param[in] numGridLevels   the number of grid levels (must be >= 1)
		 */
		StructuredMultiGrid
		(
			const CoordVector<dim>& vLowerBounds,
			const CoordVector<dim>& vUpperBounds,
			const CoordVector<dim, std::size_t>& vNumElemPerDim,
			const std::size_t numGridLevels,
			mpi::Comm comm
		);

		/// refine the hierarchy
		/**
		 * Add new grid level with a halved mesh size w.r.t. to the finest current grid
		 * This operation increases the number of grid levels in the MultiGrid by one.
		 */
		void refine();

		/// number of vertices in the grid
		std::size_t num_levels() const;

		/// returns a reference to the grid on a level (numbered: 0, 1, ..., num_levels() - 1)
		const StructuredGrid<dim>& grid(std::size_t lev) const;

		/// vector of lower grid bounds
		const CoordVector<dim>& lower_bnds() const;

		/// vector of upper grid bounds
		const CoordVector<dim>& upper_bnds() const;

	private:
		std::vector<StructuredGrid<dim> > m_vSGrid;

		CoordVector<dim> m_vLowerBounds;
		CoordVector<dim> m_vUpperBounds;
		CoordVector<dim, std::size_t> m_vNumElemPerDimBaseLvl;
};

#endif // STRUCTURED_MULTI_GRID_H
