/*
 * poisson_disc.h
 *
 * author: a.vogel@rub.de
 *
 */

#include "grid/structured_grid.h"
#include "algebra/vector.h"
#include "assemble_interface.h"
#include "algebra/lil_matrix.h"
#include "algebra/ell_matrix.h"
#include "algebra/vector.h"
#include <iostream>
#include <ctime>

template <typename TMatrix, std::size_t dim>
void AssemblePoissonMatrix
(
	TMatrix& mat,
	Vector<double>& sol,
	Vector<double>& rhs,
	const StructuredGrid<dim>& rGrid,
	bool (*DirichletBndCnd)(double& val, const CoordVector<dim>& coords),
	double (*Rhs)(const CoordVector<dim>& coords)
)
{
	// clear and resize matrix
	std::size_t nVrt = rGrid.num_vertices();
	mat.reinit(nVrt, nVrt);
	rhs.clear();
	rhs.resize(nVrt, 0.0);
	sol.resize(nVrt);

	///////////////////////////////////////
	// inner couplings and right-hand side
	///////////////////////////////////////

	// number of vertices
	std::size_t numVertexPerElem = 1;
	for(std::size_t d = 0; d < dim; ++d) numVertexPerElem *= 2;

	// mesh size
	const CoordVector<dim>& h = rGrid.mesh_sizes();

	// right-hand side is integrated by midpoint rule
	// we compute the volume fraction of each element (line, quad, hex)
	// that contributes to one of the nodes of the element
	double volume = 1;
	for(std::size_t d = 0; d < dim; ++d)
		volume *= (h[d] / 2.0);

	// matrix is assembled by finite-element using a tensor-product stencil
	// approach. This corresponds for 1d (lines) and 2d (structured triangles)
	// to a finite element assemblings of linear elements
	// 3d case is handled analogously
	CoordVector<dim, double> ShapeProdIntegrated;
	// set derivate
	for(std::size_t d = 0; d < dim; ++d)
		ShapeProdIntegrated[d] = (1 / h[d]) * (1 / h[d]);
	// scale by integration domain
	for(std::size_t d = 0; d < dim; ++d){
		ShapeProdIntegrated[d] *= (1./dim);
		for(std::size_t d2 = 0; d2 < dim; ++d2)
			ShapeProdIntegrated[d] *= h[d2];
	}

	// loop elements
	std::size_t nElem = rGrid.num_elements();
	for (std::size_t elem = 0; elem < nElem; ++elem)
	{
		CoordVector<dim, std::size_t> elemMultiIndex = rGrid.element_multi_index(elem);

		// loop each vertex of the element
		for(std::size_t vrt = 0; vrt < numVertexPerElem; ++vrt){

			// compute offsets
			std::size_t ind = vrt;		
			CoordVector<dim, std::size_t> addMultiIndex;
			for (std::size_t i = 0; i < dim - 1; ++i)
			{
				addMultiIndex[i] = ind % 2;
				ind = (ind - addMultiIndex[i]) / 2;
			}
			addMultiIndex[dim - 1] = ind;

			// from vertex
			CoordVector<dim, std::size_t> fromMultiIndex;
			for(std::size_t d = 0; d < dim; ++d)
				fromMultiIndex[d] = elemMultiIndex[d] + addMultiIndex[d];
			std::size_t vrtFrom = rGrid.vertex(fromMultiIndex);

			// loop coupled neigbours, compute matrix contribution
			// every vertex has exactly dim neigbours it is connected to (locally on the element)
			for(std::size_t nbr = 0; nbr < dim; ++nbr){

				// to vertex
				// start at fromVertex and move in one of the dimensions
				CoordVector<dim, std::size_t> toMultiIndex;
				for(std::size_t d = 0; d < dim; ++d) toMultiIndex[d] = fromMultiIndex[d];
				toMultiIndex[nbr] = elemMultiIndex[nbr] + ((addMultiIndex[nbr] + 1) % 2);		
				std::size_t vrtTo = rGrid.vertex(toMultiIndex);

				// add matrix contributions
				mat(vrtFrom, vrtFrom) += ShapeProdIntegrated[nbr];
				mat(vrtFrom, vrtTo) -= ShapeProdIntegrated[nbr];
			}

			// compute rhs contribution
			CoordVector<dim> coords;
			rGrid.vertex_coords(coords, vrtFrom);
			rhs[vrtFrom] += Rhs(coords) * volume;
		}
	}

	//////////////////////////////////////////////////
	// set dirichlet values, modify right-hand side
	//////////////////////////////////////////////////

	// random init values
//	std::srand(std::time(0));

	double bndValue;
	CoordVector<dim> coords;
	for (std::size_t vrt = 0; vrt < nVrt; ++vrt)
	{
		rGrid.vertex_coords(coords, vrt);
		if (rGrid.is_boundary(vrt) && DirichletBndCnd(bndValue, coords))
		{
			std::vector<std::size_t> vNeighbors;
			rGrid.vertex_neighbors(vNeighbors, vrt);
			for (size_t n = 0; n < vNeighbors.size(); ++n)
			{
				std::size_t neighbor = vNeighbors[n];

				mat(vrt, neighbor) = 0;
				rhs[neighbor] -= mat(neighbor, vrt) * bndValue;
				mat(neighbor, vrt) = 0;
			}
			mat(vrt, vrt) = 1.0;
			rhs[vrt] = bndValue;
			sol[vrt] = bndValue;
		}
		else
		{
			sol[vrt] = 0;
// note: using random initialization does not produce a consistent vector
//			sol[vrt] = std::rand() / (double) RAND_MAX;
		}
	}

	rhs.set_storage_type(PST_ADDITIVE);
	rhs.set_layouts(rGrid.layouts());

	sol.set_storage_type(PST_CONSISTENT);
	sol.set_layouts(rGrid.layouts());
}




/** @brief Finite element discretization for the Poisson equation.
 * This class assembles matrix and rhs for a finite element
 * discretization of the Poisson equation on a structured grid.
 *
 * The Poisson equation reads:
 *     -laplace u = f,
 * where u is the unknown function and f is a known function ("rhs").
 * A Dirichlet condition,
 *     u = g on some part of the boundary,
 * where g is a known function, allows for a unique solution to this problem.
 *
 * Both right-hand side (f) and Dirichlet value (g) are provided to this
 * implementation via function pointers.
 */
template <std::size_t dim>
class PoissonDisc : public IAssemble<dim>
{
	public:
		/// dirichlet boundary function type
		typedef bool (*DirichletBndCnd)(double& value, const CoordVector<dim>& coords);

		/// right-hand side function type
		typedef double (*Rhs)(const CoordVector<dim>& coords);

	public:
		/// Constructor
		PoissonDisc() {};

		/// set the Dirichlet boundary function
		void set_dirichlet_boundary(const DirichletBndCnd DiriBndFct){ m_pDiriBndFct = DiriBndFct;}

		/// set the right-hand side function
		void set_rhs(const Rhs RhsFct) {m_pRhsFct = RhsFct;}

		/// assemble matrix and rhs vector for the discretization
		void assemble(ELLMatrix<double>& mat, Vector<double>& rhs, Vector<double>& u, const StructuredGrid<dim>& grid) const
		{
				LILMatrix<double> lilA;
				AssemblePoissonMatrix<LILMatrix<double>, dim>(lilA, u, rhs, grid, m_pDiriBndFct, m_pRhsFct);
				ELLMatrix<double> A(lilA);
				mat = A;
				mat.set_storage_type(PST_ADDITIVE);
				mat.set_layouts(u.layouts());
		}


		/// assemble matrix for the discretization
		void assemble(ELLMatrix<double>& mat, Vector<double>& u, const StructuredGrid<dim>& grid) const
		{
				LILMatrix<double> lilA;
				Vector<double> rhs;
				AssemblePoissonMatrix<LILMatrix<double>, dim>(lilA, u, rhs, grid, m_pDiriBndFct, m_pRhsFct);
				ELLMatrix<double> A(lilA);
				mat = A;			
				mat.set_storage_type(PST_ADDITIVE);
				mat.set_layouts(u.layouts());
		}


	private:
		DirichletBndCnd m_pDiriBndFct;
		Rhs m_pRhsFct;
};

