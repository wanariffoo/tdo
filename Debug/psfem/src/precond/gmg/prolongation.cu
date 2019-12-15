/*
 * prolongation.cpp
 *
 * author: a.vogel@rub.de
 *
 */

#include "prolongation.h"
#include <iostream>
#include "algebra/vector.h"
#include "algebra/ell_matrix.h"

template <std::size_t dim>
Prolongation<dim>::Prolongation()
: m_diriBndFct(NULL)
{}


template <std::size_t dim>
void Prolongation<dim>::set_dirichlet_boundary(const DirichletBndFct fct)
{
	m_diriBndFct = fct;
}


template <std::size_t dim>
void Prolongation<dim>::
assemble(	ELLMatrix<double>& ellmat,
			const StructuredGrid<dim>& fineGrid,
			const StructuredGrid<dim>& coarseGrid) const
{
	LILMatrix<double> mat;

	// clear matrix and resize
	mat.reinit(fineGrid.num_vertices(), coarseGrid.num_vertices(), 0.0);

	// loop coarse vertices
	for (std::size_t cvrt = 0; cvrt < coarseGrid.num_vertices(); ++cvrt)
	{
		// get coarse grid multi index and coords
		CoordVector<dim, std::size_t> vCoarseMultiIndex = coarseGrid.vertex_multi_index(cvrt);

		// compute corresponding fine multi index
		CoordVector<dim, std::size_t> vFineMultiIndex;
		for(std::size_t d = 0; d < dim; ++d){
			vFineMultiIndex[d] = vCoarseMultiIndex[d] * 2;
		}

		// on Dirichlet boundary vertices, assemble only Dirichlet row/cols
		double bndVal;
		CoordVector<dim> cvrtCoords;
		coarseGrid.vertex_coords(cvrtCoords, cvrt);
		if (coarseGrid.is_boundary(cvrt) && m_diriBndFct && m_diriBndFct(bndVal, cvrtCoords)){

			// get fine index
			std::size_t fvrt = fineGrid.vertex(vFineMultiIndex);

			// only direct child
			mat(fvrt, cvrt) = 1.0;

			// others not coupled
			continue;
		}

		// loop all fine neighbors
		int vOffset[dim];
		for(std::size_t d = 0; d < dim; ++d) vOffset[d] = -1;
		CoordVector<dim, std::size_t> vNeighborMultiIndex;

		bool done = false;
		while(!done){


			bool valid = true;
			double scale = 1;
			for(std::size_t d = 0; d < dim; ++d){

				const int index = vFineMultiIndex[d] + vOffset[d];
				if( (index < 0) || (index >= (int)fineGrid.num_vertices_per_dim()[d])){
					valid = false;
				} else{
					vNeighborMultiIndex[d] = index;
				}

				if(vOffset[d] != 0)
					scale *= 0.5;
			}

			if(valid) {
				std::size_t fvrt = fineGrid.vertex(vNeighborMultiIndex);
				mat(fvrt, cvrt) = scale;
			}

			vOffset[0]++;
			std::size_t dimCnt = 0;
			while(vOffset[dimCnt] > 1){
				if(dimCnt == dim - 1){
					done = true;
					break;
				}
				vOffset[dimCnt++] = -1;
				vOffset[dimCnt]++;
			}
		}
	}

	// convert to ellpack
	ellmat = mat;

//	std::cout << ellmat;
}



template <std::size_t dim>
void Prolongation<dim>::interpolate
(
	Vector<double>& fineSol,
	const Vector<double>& coarseSol,
	const StructuredGrid<dim>& fineGrid,
	const StructuredGrid<dim>& coarseGrid
) const
{
	// resize fine solution
	fineSol.clear();
	fineSol.resize(fineGrid.num_vertices(), 0.0);

	// loop coarse vertices
	std::size_t nC = coarseGrid.num_vertices();
	for (std::size_t cvrt = 0; cvrt < nC; ++cvrt)
	{
		// get coarse grid multi index and coords
		CoordVector<dim, std::size_t> vCoarseMultiIndex = coarseGrid.vertex_multi_index(cvrt);

		// compute corresponding fine multi index
		CoordVector<dim, std::size_t> vFineMultiIndex;
		vFineMultiIndex = vCoarseMultiIndex * (std::size_t) 2;

		// loop all fine neighbors
		int vOffset[dim];
		for(std::size_t d = 0; d < dim; ++d) vOffset[d] = -1;
		CoordVector<dim, std::size_t> vNeighborMultiIndex;

		bool done = false;
		while (!done)
		{
			bool valid = true;
			double scale = 1;
			for(std::size_t d = 0; d < dim; ++d)
			{
				const int index = vFineMultiIndex[d] + vOffset[d];
				if ((index < 0) || (index >= (int)fineGrid.num_vertices_per_dim()[d]))
					valid = false;
				else
					vNeighborMultiIndex[d] = index;

				if (vOffset[d] != 0)
					scale *= 0.5;
			}

			if (valid)
			{
				std::size_t fvrt = fineGrid.vertex(vNeighborMultiIndex);
				fineSol[fvrt] += scale * coarseSol[cvrt];
			}

			vOffset[0]++;
			std::size_t dimCnt = 0;
			while (vOffset[dimCnt] > 1)
			{
				if (dimCnt == dim - 1)
				{
					done = true;
					break;
				}
				vOffset[dimCnt++] = -1;
				vOffset[dimCnt]++;
			}
		}
	}

	fineSol.set_storage_type(coarseSol.get_storage_mask());
	fineSol.set_layouts(fineGrid.layouts());
}


// explicit template declarations
template class Prolongation<1>;
template class Prolongation<2>;
template class Prolongation<3>;
