/*
 * assemble_interface.h
 *
 * author: a.vogel@rub.de
 *
 */

#ifndef ASSEMBLE_INTERFACE_H
#define ASSEMBLE_INTERFACE_H

#include <cstddef>
#include <iostream>
#include "algebra/vector.h"
#include "algebra/ell_matrix.h"
#include "grid/structured_grid.h"



/// interface class for discretizations
template <std::size_t dim>
class IAssemble
{
	public:
		/// constructor
		IAssemble(){};

		/// destructor
		virtual ~IAssemble() {};

		/** @brief Assemble matrix and right-hand side
		 *
		 * @param mat    matrix to be filled
		 * @param rhs    vector to be filled
		 * @param u      solution to be used
		 * @param grid   grid used for assembling
		 */
		virtual void assemble
		(
			ELLMatrix<double>& mat,
			Vector<double>& rhs,
			Vector<double>& u,
			const StructuredGrid<dim>& grid
		) const = 0;

		/** @brief Assemble matrix
		 *
		 * @param mat    matrix to be filled
		 * @param u      solution to be used
		 * @param grid   grid used for assembling
		 */
		virtual void assemble
		(
			ELLMatrix<double>& mat,
			Vector<double>& u,
			const StructuredGrid<dim>& grid
		) const = 0;

};


#endif // ASSEMBLE_INTERFACE_H
