/*
 * structured_grid.h
 *
 * author: a.vogel@rub.de
 *
 */

#ifndef STRUCTURED_GRID_H
#define STRUCTURED_GRID_H

#include "util/coord_vector.h"
#include "parallel/parallel.h"
#include "parallel/layout.h"

#include <cstddef>
#include <vector>
#include <memory>


/**
 * A simple structured grid.
 */
template <std::size_t dim>
class StructuredGrid
{
	public:
		StructuredGrid() {};

		/// constructor with bounds and resolution
		StructuredGrid
		(
			const CoordVector<dim>& vLowerBounds,
			const CoordVector<dim>& vUpperBounds,
			const CoordVector<dim, std::size_t>& vNumElemPerDim,
			mpi::Comm comm
		);


		/// number of vertices in the grid
		std::size_t num_vertices() const;

		/// number of elements in the grid
		std::size_t num_elements() const;

		/// coordinates of an index
		void vertex_coords(CoordVector<dim>& coordsOut, std::size_t vrt) const;

		/// vertices of an element
		void vertices_of_element(std::vector<std::size_t>& vrtsOut, std::size_t elem) const;

		/// index neighbors
		void vertex_neighbors(std::vector<std::size_t>& vNeighborIndex, std::size_t vrt) const;

		/// returns if a vertex index is located on the grid boundary
		bool is_boundary(std::size_t vrt) const;

		/// lower bound
		const CoordVector<dim>& lower_bnds() const;

		/// upper bound
		const CoordVector<dim>& upper_bnds() const;

		/// number of vertices for each dimension
		const CoordVector<dim, std::size_t>& num_vertices_per_dim() const;

		/// number of elements for each dimension
		const CoordVector<dim, std::size_t>& num_elements_per_dim() const;

		/// mesh size (distance of adjacent vertices) in each dimension
		const CoordVector<dim>& mesh_sizes() const;

	public:
		/// returns multi-index in lexicographic numbering, i.e. numbered dimension by dimension
		CoordVector<dim, std::size_t> vertex_multi_index(std::size_t vrt) const;

		/// returns index corresponding to multi_index numbering
		std::size_t vertex(const CoordVector<dim, std::size_t>& vMultiIndex) const;


		/// returns multi-index in lexicographic numbering, i.e. numbered dimension by dimension
		CoordVector<dim, std::size_t> element_multi_index(std::size_t elem) const;

		/// returns index corresponding to multi_index numbering
		std::size_t element(const CoordVector<dim, std::size_t>& vMultiIndex) const;

	public:
		std::shared_ptr<AlgebraLayout> layouts() const {return m_spAlgebraLayout;}

	protected:
		/// create interfaces
		void create_interfaces(AlgebraLayout& layouts) const;

	protected:
		CoordVector<dim> m_vLowerBnds;
		CoordVector<dim> m_vUpperBnds;
		CoordVector<dim, std::size_t> m_vNumElemPerDim;
		CoordVector<dim, std::size_t> m_vNumVertexPerDim;
		CoordVector<dim> m_vMeshSize;

		std::shared_ptr<AlgebraLayout> m_spAlgebraLayout;
};


#endif // STRUCTURED_GRID_H
