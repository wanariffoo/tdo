
#ifndef VTK_H
#define VTK_H

#include <string>
#include <vector>

#include <fstream>
#include <stdexcept>
#include <sstream>


// template <std::size_t dim>
// void WriteVectorToVTK(	const Vector<double>& vec, const StructuredGrid<dim>& m_grid,
// 						const std::string& fctName, const std::string& filename)



// TODO: include displacements here too
// TODO: displacements mustn't be SCALARS, but VECTORS
// outputs design variable vector
void WriteVectorToVTK(vector<double> &chi, vector<double> &u, const std::string& filename, size_t dim, vector<size_t> numNodesPerDim, double h, size_t numElements, size_t numNodes)
{

	std::ofstream ofs(filename, std::ios::out);
	if (ofs.bad())
	{
		std::ostringstream oss;
		oss << "File '" << filename << "' could not be opened for writing.";
		throw std::runtime_error(oss.str());
	}

	ofs << "# vtk DataFile Version 2.0" << std::endl;
	ofs << "Thermodynamics Topology Optimzation" << std::endl;
	ofs << "ASCII" << std::endl;
	ofs << endl;
	ofs << "DATASET STRUCTURED_GRID" << std::endl;

	if ( dim == 2 )
		numNodesPerDim.push_back(1);

	// specify number of nodes in each dimension
	ofs << "DIMENSIONS";
	for (std::size_t i = 0; i < 3; ++i)
		ofs << " " << numNodesPerDim[i];
	// for (std::size_t i = dim; i < 3; ++i)
	// 		ofs << " " << 1;
	ofs << std::endl;

	// specify the coordinates of all points
	ofs << "POINTS ";
	ofs << numNodes << " float" << endl;

	for (std::size_t z = 0; z < numNodesPerDim[2]; ++z)
	{
		for (std::size_t y = 0; y < numNodesPerDim[1]; ++y)
		{
			for (std::size_t x = 0; x < numNodesPerDim[0]; ++x)
				ofs << " " << h*x << " " << h*y << " " << h*z << endl;
		}
	}

	ofs << endl;

	// specifying the design variable in each element
	ofs << "CELL_DATA " << numElements << endl;
	ofs << "SCALARS chi double" << endl;
	ofs << "LOOKUP_TABLE default" << endl;

	for (int i = 0 ; i < numElements ; i++)
		ofs << " " << chi[i] << endl;

	ofs << endl;

	// specifying the displacements for all dimensions in each point
	ofs << "POINT_DATA " << numNodes << std::endl;
	ofs << "VECTORS displacements double" << std::endl;

	
	for ( int i = 0 ; i < numNodes ; i++ )
	{
		if ( dim == 2 )
		{
			for ( int j = 0 ; j < 2 ; j++ )
				ofs << " " << u[dim*i + j];

			// setting displacement in z-dimension to zero
			ofs << " 0";
		}

		else
		{
			for ( int j = 0 ; j < dim ; j++ )
				ofs << " " << u[dim*i + j];
		}

		ofs << endl;
	}


	// ofs << "LOOKUP_TABLE default" << std::endl;

	// // now loop data in vector
	// std::size_t n = vec.size();
	// // assert(n == m_grid.num_vertices() && "Size mismatch: #vector entries != # mesh vertices.");
	// for (std::size_t i = 0; i < n; ++i)
	// 	ofs << vec[i] << std::endl;

	// ofs.close();
	
}

// // outputs displacement vector
// void WriteVectorToVTK(vector<double> &vec, const std::string& fctName, const std::string& filename, size_t dim, vector<size_t> N, double h, size_t numElements )
// {

// 	std::ofstream ofs(filename, std::ios::out);
// 	if (ofs.bad())
// 	{
// 		std::ostringstream oss;
// 		oss << "File '" << filename << "' could not be opened for writing.";
// 		throw std::runtime_error(oss.str());
// 	}

// 	ofs << "# vtk DataFile Version 2.0" << std::endl;
// 	ofs << "Thermodynamics Topology Optimzation" << std::endl;
// 	ofs << "ASCII" << std::endl;
// 	ofs << "DATASET STRUCTURED_POINTS" << std::endl;


// 	ofs << "DIMENSIONS";
// 	// const CoordVector<dim, std::size_t>& meshSz = m_grid.num_vertices_per_dim();
// 	for (std::size_t i = 0; i < dim; ++i)
// 		ofs << " " << N[i];
// 	for (std::size_t i = dim; i < 3; ++i)
// 			ofs << " " << 1;
// 	ofs << std::endl;

    
// 	ofs << "SPACING";
// 	// const CoordVector<dim>& elemSz = m_grid.mesh_sizes();
// 	for (std::size_t i = 0; i < dim; ++i)
// 		ofs << " " << h;
// 	for (std::size_t i = dim; i < 3; ++i)
// 			ofs << " " << 1.0;
// 	ofs << std::endl;


// 	ofs << "ORIGIN";
// 	// const CoordVector<dim>& loBnds = m_grid.lower_bnds();
// 	for (std::size_t i = 0; i < dim; ++i)
// 		ofs << " " << 0;
// 	for (std::size_t i = dim; i < 3; ++i)
// 			ofs << " " << 0.0;
// 	ofs << std::endl;

// 	ofs << "POINT_DATA " << numElements << std::endl;
// 	ofs << "SCALARS " << fctName << " double 1" << std::endl;


// 	ofs << "LOOKUP_TABLE default" << std::endl;

// 	// now loop data in vector
// 	std::size_t n = vec.size();
// 	// assert(n == m_grid.num_vertices() && "Size mismatch: #vector entries != # mesh vertices.");
// 	for (std::size_t i = 0; i < n; ++i)
// 		ofs << vec[i] << std::endl;

// 	ofs.close();
	
// }


#endif // VTK_H
