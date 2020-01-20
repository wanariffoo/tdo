
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


// template <std::size_t dim>
void WriteVectorToVTK(vector<double> &vec, const std::string& fctName, const std::string& filename, size_t dim, vector<size_t> meshSz, double elemSz, size_t numNodes )
{
    cout << "writeVectorToVTK" << endl;
	std::ofstream ofs(filename, std::ios::out);
	if (ofs.bad())
	{
		std::ostringstream oss;
		oss << "File '" << filename << "' could not be opened for writing.";
		throw std::runtime_error(oss.str());
	}

	ofs << "# vtk DataFile Version 2.0" << std::endl;
	ofs << "Exported solution" << std::endl;
	ofs << "ASCII" << std::endl;
	ofs << "DATASET STRUCTURED_POINTS" << std::endl;

	ofs << "DIMENSIONS";
	// const CoordVector<dim, std::size_t>& meshSz = m_grid.num_vertices_per_dim();
	for (std::size_t i = 0; i < dim; ++i)
		ofs << " " << meshSz[i];
	for (std::size_t i = dim; i < 3; ++i)
			ofs << " " << 1;
	ofs << std::endl;

    
	ofs << "SPACING";
	// const CoordVector<dim>& elemSz = m_grid.mesh_sizes();
	for (std::size_t i = 0; i < dim; ++i)
		ofs << " " << elemSz;
	for (std::size_t i = dim; i < 3; ++i)
			ofs << " " << 1.0;
	ofs << std::endl;



	ofs << "ORIGIN";
	// const CoordVector<dim>& loBnds = m_grid.lower_bnds();
	for (std::size_t i = 0; i < dim; ++i)
		ofs << " " << 0;
	for (std::size_t i = dim; i < 3; ++i)
			ofs << " " << 0.0;
	ofs << std::endl;

	ofs << "POINT_DATA " << numNodes << std::endl;
	ofs << "SCALARS " << fctName << " double 1" << std::endl;

    cout << "numNodes = " << numNodes << endl;

	ofs << "LOOKUP_TABLE default" << std::endl;

	// now loop data in vector
	std::size_t n = vec.size();
	// assert(n == m_grid.num_vertices() && "Size mismatch: #vector entries != # mesh vertices.");
	for (std::size_t i = 0; i < n; ++i)
		ofs << vec[i] << std::endl;

	ofs.close();
}


#endif // VTK_H
