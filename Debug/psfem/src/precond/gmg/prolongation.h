/*
 * prolonagtion.h
 *
 * author: a.vogel@rub.de
 *
 */

#ifndef PROLONGATION_H
#define PROLONGATION_H

#include "util/coord_vector.h"
#include "grid/structured_grid.h"
#include "algebra/vector.h"
#include "algebra/ell_matrix.h"
#include <cstddef>
#include <iostream>



/// Interface class for prolongation assembling
template <std::size_t dim>
class IProlongation
{
	public:
		/// constructor
		IProlongation(){};

		/// destructor
		virtual ~IProlongation() {};

		/** @brief Assemble prolongation matrix
		 *
		 * @param mat    		matrix to be filled
		 * @param fineGrid  	fine grid used for assembling
		 * @param coarseGrid  	coarse grid used for assembling
		 */
		virtual void assemble(	ELLMatrix<double>& mat,
								const StructuredGrid<dim>& fineGrid,
								const StructuredGrid<dim>& coarseGrid) const = 0;

		/// direct and full prolongation without respect to Dirichlet bnd
		virtual void interpolate
		(
			Vector<double>& fineSol,
			const Vector<double>& coarseSol,
			const StructuredGrid<dim>& fineGrid,
			const StructuredGrid<dim>& coarseGrid
		) const
		{
			std::cout << "Direct interpolation not implemented." << std::endl;
			throw 1;
		}
};



/** @brief Prolongation for a structured grid hierarchy
 * This class assembles the prolonagtion matrix
 * transfering the grid grid vector between two structured grids.
 *
 * In order to handle Dirichlet values, these are provided to this
 * implementation via function pointers.
 */
template <std::size_t dim>
class Prolongation
 : public IProlongation<dim>
{
	public:
		/// Type declaration for the Dirichlet boundary function pointer type
		typedef bool (*DirichletBndFct)(double& value, const CoordVector<dim>& coords);

	public:
		/// Constructor
		Prolongation();

		/// set the Dirichlet boundary function
		void set_dirichlet_boundary(const DirichletBndFct fct);

		/// Assemble prolonagtion matrix
		virtual void assemble(	ELLMatrix<double>& mat,
								const StructuredGrid<dim>& fineGrid,
								const StructuredGrid<dim>& coarseGrid) const;

		/// direct and full prolongation without respect to Dirichlet bnd
		void interpolate
		(
			Vector<double>& fineSol,
			const Vector<double>& coarseSol,
			const StructuredGrid<dim>& fineGrid,
			const StructuredGrid<dim>& coarseGrid
		) const;

	protected:
		DirichletBndFct m_diriBndFct;
};

#ifdef CUDA
	#include "prolongation.cu"
#endif


#endif // PROLONGATION_H
