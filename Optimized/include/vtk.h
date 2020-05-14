/*
	vtk.h
	
    Developed for the master thesis project: GPU-accelerated Thermodynamic Topology Optimization
    Author: Wan Arif bin Wan Abhar
    Institution: Ruhr Universitaet Bochum
*/

#ifndef VTK_H
#define VTK_H

#include <string>
#include <vector>

#include <fstream>
#include <stdexcept>
#include <sstream>



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

	ofs << std::endl;

	// specify the coordinates of all points
	ofs << "POINTS ";
	ofs << numNodes << " float" << endl;

	if ( dim == 2 )
	{
		for (std::size_t z = 0; z < numNodesPerDim[2]; ++z)
		{
			for (std::size_t y = 0; y < numNodesPerDim[1]; ++y)
			{
				for (std::size_t x = 0; x < numNodesPerDim[0]; ++x)
					ofs << " " << h*x << " " << h*y << " " << h*z << endl;
			}
		}
	}


	else // dim == 3
	{
		for (std::size_t z = 0; z < numNodesPerDim[2]; ++z)
		{
			for (std::size_t y = 0; y < numNodesPerDim[1]; ++y)
			{
				for (std::size_t x = 0; x < numNodesPerDim[0]; ++x)
					ofs << " " << h*x << " " << h*y << " " << (-h*z) << endl;
			}
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
	
}

#endif // VTK_H
