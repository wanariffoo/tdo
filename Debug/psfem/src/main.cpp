/*
 * main.cpp
 *
 * author: a.vogel@rub.de
 *
 */

#include "util/coord_vector.h"
#include "util/vtk.h"

#include "grid/structured_grid.h"
#include "grid/structured_multi_grid.h"
#include "disc/poisson_disc.h"

#include "algebra/vector.h"
#include "algebra/dense_matrix.h"
#include "algebra/lil_matrix.h"
#include "algebra/ell_matrix.h"

#include "solver/iterative_solver.h"
#include "solver/cg.h"

#include "precond/jacobi.h"
#include "precond/gs.h"
#include "precond/ilu.h"
#include "precond/richardson.h"

#include "precond/gmg/gmg.h"
#include "precond/gmg/gmg_nested.h"
#include "precond/gmg/prolongation.h"

#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <chrono>
#include <cstdlib>
#include <cmath>
const double PI  = 3.141592653589793238463;
using namespace std;


////////////////////////////////////
// user data functions
////////////////////////////////////

// boundary value problems
enum class Problem {
	CONSTANT,
	LINEAR,
	QUADRATIC,
	ADDITIVE_SINUS,
	MULTIPLICATIVE_SINUS
};

// select a boundary value problem
const Problem problem = Problem::MULTIPLICATIVE_SINUS;


// (known) Solution for Poisson problem
template <std::size_t dim>
double Solution(const CoordVector<dim>& x)
{
	switch(problem){
		case Problem::ADDITIVE_SINUS:
		{
			const double s = 2*PI;
			double val = 0.0;
			for (std::size_t i = 0; i < dim; ++i)
				val += sin(s * x[i]);
			return val;
		}

		case Problem::MULTIPLICATIVE_SINUS:
		{
			const double s = 2*PI;
			double val = 1.0;
			for (std::size_t i = 0; i < dim; ++i)
				val *= sin(s * x[i]);
			return val;
		}

		case Problem::QUADRATIC:
		{
			double val = dim;
			for (std::size_t i = 0; i < dim; ++i)
				val -= x[i]*x[i];
			return val;
		}

		case Problem::LINEAR:
		{
			double val = 0;
			for (std::size_t i = 0; i < dim; ++i)
				val += x[i];
			return val;
		}

		case Problem::CONSTANT:
		{
			return 5.0;
		}

		default: throw(std::runtime_error("boundary value problem unknown"));
	}
}

// Right-hand side function for Poisson problem
template <std::size_t dim>
double Rhs(const CoordVector<dim>& x)
{
	switch(problem){
		case Problem::ADDITIVE_SINUS:
		{
			const double s = 2*PI;
			double val = 0.0;
			for (std::size_t i = 0; i < dim; ++i)
				val += sin(s * x[i]);
			return s*s*val;
		}

		case Problem::MULTIPLICATIVE_SINUS:
		{
			const double s = 2*PI;
			double val = 1.0;
			for (std::size_t i = 0; i < dim; ++i)
				val *= sin(s * x[i]);
			return 2*s*s*val;
		}

		case Problem::QUADRATIC:
		{
			return 2.0 * dim;
		}

		case Problem::LINEAR:
		{
			return 0.0;
		}

		case Problem::CONSTANT:
		{
			return 0.0;
		}

		default: throw(std::runtime_error("boundary value problem unknown"));
	}
}

// Dirichlet boundary for Poisson problem
template <std::size_t dim>
bool DirichletBndCnd(double& val, const CoordVector<dim>& x)
{
	// check if actually bnd node
	const double eps = 1e-8;
	bool retFlag = false;
	for (std::size_t i = 0; i < dim; ++i){
		retFlag |= (fabs(x[i]) < eps) || (fabs(x[i] - 1.0) < eps);
	}
	if(!retFlag) return false;

	// compute boundary value
	val = Solution(x);
	return true;
}


////////////////////////////////////
// main program
////////////////////////////////////

int main(int argc, char** argv) 
{
	mpi::init(argc, argv);

	const int dim = 2;

	///////////////////////////////////////////
	// Create grid
	///////////////////////////////////////////

	// structured mesh with upper and lower bound, number of grid points
	const CoordVector<dim> ll(0.0), ur(1.0);
	const int N = 8; 
	const int numLevel = 2;
	CoordVector<dim, std::size_t> vNumElemPerDim(N);
	StructuredMultiGrid<dim> mg(ll, ur, vNumElemPerDim, numLevel, mpi::world());
	const StructuredGrid<dim>& grid = mg.grid(mg.num_levels()-1);

	///////////////////////////////////////////
	// Assemble Matrix and right-hand side
	///////////////////////////////////////////

	// create Vectors x,b
	Vector<double> x, b;

	// start time measurement for assembling
	mpi::world().barrier(); double startAssemble = mpi::walltime();

	// assemble matrix
	LILMatrix<double> lilA;
	AssemblePoissonMatrix<LILMatrix<double>, dim>(lilA, x, b, grid, DirichletBndCnd, Rhs);
	ELLMatrix<double> A(lilA, x.layouts());
	A.set_storage_type(PST_ADDITIVE);

	// stop time measurement for assembling
	mpi::world().barrier(); double stopAssemble = mpi::walltime();

	// output 
	if(mpi::world().rank() == 0){
		cout << "Size per process: " << endl;
		cout << "Vector: size = " << x.size() << endl;
		cout << "Matrix: rows = " << A.num_rows() << ", cols = " << A.num_cols() << ", nnz = " << A.nnz() << endl;
	}

	///////////////////////////////////////////
	// create GMG (precond)
	///////////////////////////////////////////

	// disc
	PoissonDisc<dim> disc;
	disc.set_dirichlet_boundary(DirichletBndCnd);
	disc.set_rhs(Rhs);

	// prolongation
	Prolongation<dim> prol;
	prol.set_dirichlet_boundary(DirichletBndCnd);

	// smoother
	Jacobi Smoother(0.66); 
//	ILU Smoother;			// note: ILU must still be parallelized
//	GaussSeidel Smoother;   // note: gauss-seidel must still be parallelized

	// base solver
	CG BaseSolver;
	BaseSolver.set_convergence_params(100000, 1e-99, 1e-10);
	BaseSolver.set_verbose(false);

	// gmg
	GMG<dim> gmg(mg, disc, prol, Smoother, BaseSolver);
	gmg.set_cycle('V');
	gmg.set_num_presmooth(3);
	gmg.set_num_postsmooth(3);
	gmg.set_base_level(0);
	gmg.set_rap(true);

	///////////////////////////////////////////
	// create simple preconds (Jac, GS, ILU, ...)
	///////////////////////////////////////////

	// setup some linear iterator
	Jacobi jac(2./3.);
//	GaussSeidel gs; 	// note: gauss-seidel must still be parallelized
//	ILU ilu; 			// note: ILU must still be parallelized

	///////////////////////////////////////////
	// create solver (linear solver, CG, ...)
	///////////////////////////////////////////

	// setup the iterative solver
	IterativeSolver linSolver;
//	CG linSolver;
	linSolver.set_linear_iterator(gmg);
	linSolver.set_convergence_params(1000, 1e-99, 1e-10);

	///////////////////////////////////////////
	// create nested GMG
	///////////////////////////////////////////

	// nested multigrid
	NestedGMG<dim> nestedGMG(mg, disc, prol, Smoother, BaseSolver);	
	nestedGMG.set_cycle('V');
	nestedGMG.set_num_presmooth(3);
	nestedGMG.set_num_postsmooth(3);
	nestedGMG.set_base_level(0);
	nestedGMG.set_level_iterations(2);


	///////////////////////////////////////////
	// solve system
	///////////////////////////////////////////

	// set initial guess to a random guess
//	x.random(-1.0,1.0);
	x = 0.0;


	// start time measurement for solver initialization
	mpi::world().barrier(); double startInitSolver = mpi::walltime();

	linSolver.init(A);
//	nestedGMG.init(A);

	// stop time measurement for solver initialization
	mpi::world().barrier(); double stopInitSolver = mpi::walltime();


	// start time measurement for solver application
	mpi::world().barrier(); double startSolver = mpi::walltime();

	linSolver.solve(x, b);
//	nestedGMG.solve(x);

	// stop time measurement for solver application
	mpi::world().barrier(); double stopSolver = mpi::walltime();


	// output timing
	if(mpi::world().rank() == 0) {
		cout << "Assembling took:  " << (stopAssemble-startAssemble) << " s" << endl;
		cout << "Solver init took: " << (stopInitSolver-startInitSolver) << " s" << endl;
		cout << "Solver application took: " << (stopSolver-startSolver)  << " s" << endl;
	}


	///////////////////////////////////////////
	// solution output
	///////////////////////////////////////////

	stringstream ss; ss << "sol_r" << mpi::world().rank() << ".vtk";
	WriteVectorToVTK(x, grid, "x", ss.str() );


	mpi::finalize();
	return 0;
}

