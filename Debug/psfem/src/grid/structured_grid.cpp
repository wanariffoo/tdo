/*
 * structured_grid.cpp
 *
 * author:  markus.breit@gcsc.uni-frankfurt.de
 *	 		a.vogel@rub.de
 * 
 */

#include "structured_grid.h"
#include <cmath>
#include <iostream>
#include <algorithm>


template <std::size_t dim>
StructuredGrid<dim>::StructuredGrid
(
	const CoordVector<dim>& vLowerBounds,
	const CoordVector<dim>& vUpperBounds,
	const CoordVector<dim, std::size_t>& vNumElemPerDim,
	mpi::Comm comm
)
{
	// check number of processes
	int np = comm.size();
	int rank = comm.rank();
	m_spAlgebraLayout.reset(new AlgebraLayout(comm));

	///////////////////////////
	// parallel case
	///////////////////////////
	if(np > 1)
	{

		double rootd = std::pow((double) np, (double) 1.0/dim);
		std::size_t root = (std::size_t) floor(rootd + 0.5);
		if (fabs((double) root - rootd) > 1e-8)
		{
			if (rank == 0)
				std::cout << "ERROR: Number of processes must be n^dim." << std::endl;
			mpi::abort(1);
		}

		// check that vNumGridPointsPerDim allows equal distribution
		for (std::size_t i = 0; i < dim; ++i){
			if (vNumElemPerDim[i] % root != 0){
				if (rank == 0){
					std::cout << "ERROR: Grid vNumElemPerDim must be chosen such that in each dimension," << std::endl;
					std::cout << "there are a multiple of n elements if the number of processes is n^dim." << std::endl;
				}
				mpi::abort(1);
			}
		}

		// now calculate the properties of this proc's sub-grid
		for (std::size_t d = 0; d < dim; ++d)
		{
			m_vMeshSize[d] = (vUpperBounds[d] - vLowerBounds[d]) / vNumElemPerDim[d];
			m_vNumElemPerDim[d] = vNumElemPerDim[d] / root;

			std::size_t n = rank % root;
			m_vLowerBnds[d] = vLowerBounds[d] + m_vNumElemPerDim[d] * n * m_vMeshSize[d];
			m_vUpperBnds[d] = m_vLowerBnds[d] + m_vNumElemPerDim[d] * m_vMeshSize[d];
			m_vNumVertexPerDim[d] = m_vNumElemPerDim[d] + 1;

			rank = (rank-n) / root;
		}

		create_interfaces(*m_spAlgebraLayout);
	}
	///////////////////////////
	// serial case
	///////////////////////////
	else
	{
		// calculate the properties of the grid
		m_vNumElemPerDim = vNumElemPerDim;
		m_vLowerBnds = vLowerBounds;
		m_vUpperBnds = vUpperBounds;
		for (std::size_t d = 0; d < dim; ++d){
			m_vMeshSize[d] = (vUpperBounds[d] - vLowerBounds[d]) / vNumElemPerDim[d];
			m_vNumVertexPerDim[d] = vNumElemPerDim[d] + 1;
		}
	}
}


template <std::size_t dim>
std::size_t StructuredGrid<dim>::num_vertices() const
{
	std::size_t n = 1;
	for (std::size_t i = 0; i < dim; ++i)
		n *= m_vNumVertexPerDim[i];
	return n;
}


template <std::size_t dim>
std::size_t StructuredGrid<dim>::num_elements() const
{
	std::size_t n = 1;
	for (std::size_t i = 0; i < dim; ++i)
		n *= m_vNumVertexPerDim[i] - 1;
	return n;
}


template <std::size_t dim>
void StructuredGrid<dim>::vertex_coords(CoordVector<dim>& coordsOut, std::size_t ind) const
{
	for (std::size_t i = 0; i < dim - 1; ++i)
	{
		std::size_t dimPos = ind % m_vNumVertexPerDim[i];
		coordsOut[i] = m_vLowerBnds[i] + dimPos * m_vMeshSize[i];
		ind = (ind - dimPos) / m_vNumVertexPerDim[i];
	}
	coordsOut[dim-1] = m_vLowerBnds[dim-1] + ind * m_vMeshSize[dim-1];
}


template <std::size_t dim>
void StructuredGrid<dim>::vertices_of_element(std::vector<std::size_t>& vrtsOut, std::size_t elem) const
{
	CoordVector<dim, std::size_t> elemMultiIndex = element_multi_index(elem);
	CoordVector<dim, std::size_t> addMultiIndex, resultMultiIndex;

	// number of vertices
	std::size_t numVertex = 1;
	for(std::size_t d = 0; d < dim; ++d) numVertex *= 2;

	// fill all vertices
	vrtsOut.resize(numVertex);
	for(std::size_t vrt = 0; vrt < numVertex; ++vrt){

		// compute offsets
		std::size_t ind = vrt;		
		for (std::size_t i = 0; i < dim - 1; ++i)
		{
			addMultiIndex[i] = ind % 2;
			ind = (ind - addMultiIndex[i]) / 2;
		}
		addMultiIndex[dim - 1] = ind;

		// compute multi index
		for(std::size_t d = 0; d < dim; ++d)
			resultMultiIndex[d] = elemMultiIndex[d] + addMultiIndex[d];

		vrtsOut[vrt] = vertex(resultMultiIndex);
	}
}

template <std::size_t dim>
CoordVector<dim, std::size_t> StructuredGrid<dim>::vertex_multi_index(std::size_t ind) const
{
	CoordVector<dim, std::size_t> vMultiIndex;

	for (std::size_t i = 0; i < dim - 1; ++i)
	{
		vMultiIndex[i] = ind % m_vNumVertexPerDim[i];
		ind = (ind - vMultiIndex[i]) / m_vNumVertexPerDim[i];
	}
	vMultiIndex[dim - 1] = ind;

	return vMultiIndex;
}

template <std::size_t dim>
std::size_t StructuredGrid<dim>::vertex(const CoordVector<dim, std::size_t>& vMultiIndex) const
{
	std::size_t ind = vMultiIndex[0];
	std::size_t nDims = 1;
	for (std::size_t i = 1; i < dim; ++i)
	{
		nDims *= m_vNumVertexPerDim[i-1];
		ind += vMultiIndex[i] * nDims;
	}

	return ind;
}



template <std::size_t dim>
CoordVector<dim, std::size_t> StructuredGrid<dim>::element_multi_index(std::size_t elem) const
{
	CoordVector<dim, std::size_t> vMultiIndex;

	for (std::size_t i = 0; i < dim - 1; ++i)
	{
		vMultiIndex[i] = elem % m_vNumElemPerDim[i];
		elem = (elem - vMultiIndex[i]) / m_vNumElemPerDim[i];
	}
	vMultiIndex[dim - 1] = elem;

	return vMultiIndex;
}

template <std::size_t dim>
std::size_t StructuredGrid<dim>::element(const CoordVector<dim, std::size_t>& vMultiIndex) const
{
	std::size_t elem = vMultiIndex[0];
	std::size_t nDims = 1;
	for (std::size_t i = 1; i < dim; ++i)
	{
		nDims *= m_vNumElemPerDim[i-1];
		elem += vMultiIndex[i] * nDims;
	}

	return elem;
}


template <std::size_t dim>
void StructuredGrid<dim>::vertex_neighbors(std::vector<std::size_t>& vNeighborIndex, std::size_t ind) const
{
	std::size_t origInd = ind;
	std::size_t dimJump = 1;
	for (std::size_t i = 0; i < dim - 1; ++i)
	{
		std::size_t dimPos = ind % m_vNumVertexPerDim[i];

		if (dimPos > 0)	// predecessor if existent
			vNeighborIndex.push_back(origInd - dimJump);
		if (dimPos < m_vNumElemPerDim[i])	// successor if existent
			vNeighborIndex.push_back(origInd + dimJump);

		dimJump *= m_vNumVertexPerDim[i];
		ind = (ind - dimPos) / m_vNumVertexPerDim[i];
	}
	if (ind > 0)	// predecessor if existent
		vNeighborIndex.push_back(origInd - dimJump);
	if (ind < m_vNumElemPerDim[dim-1])	// successor if existent
		vNeighborIndex.push_back(origInd + dimJump);
}


template <std::size_t dim>
bool StructuredGrid<dim>::is_boundary(std::size_t ind) const
{
	for (std::size_t i = 0; i < dim - 1; ++i)
	{
		std::size_t dimPos = ind % m_vNumVertexPerDim[i];
		if (dimPos == 0 || dimPos == m_vNumElemPerDim[i])
			return true;
		ind = (ind - dimPos) / m_vNumVertexPerDim[i];
	}
	return (ind == 0 || ind == m_vNumElemPerDim[dim-1]);
}


template <std::size_t dim>
const CoordVector<dim>& StructuredGrid<dim>::lower_bnds() const
{
	return m_vLowerBnds;
}


template <std::size_t dim>
const CoordVector<dim>& StructuredGrid<dim>::upper_bnds() const
{
	return m_vUpperBnds;
}


template <std::size_t dim>
const CoordVector<dim, std::size_t>& StructuredGrid<dim>::num_vertices_per_dim() const
{
	return m_vNumVertexPerDim;
}

template <std::size_t dim>
const CoordVector<dim, std::size_t>& StructuredGrid<dim>::num_elements_per_dim() const
{
	return m_vNumElemPerDim;
}


template <std::size_t dim>
const CoordVector<dim>& StructuredGrid<dim>::mesh_sizes() const
{
	return m_vMeshSize;
}



template <std::size_t dim>
void StructuredGrid<dim>::create_interfaces(AlgebraLayout& layouts) const
{
	// prepare layouts
	Layout& masterLayout = layouts.master_layout();
	Layout& slaveLayout = layouts.slave_layout();
	masterLayout.clear();
	slaveLayout.clear();

	// calculate the coords of this proc in the procGrid
	std::size_t procInd = layouts.comm().rank();
	std::size_t np = layouts.comm().size();

	std::size_t dimProcs = (std::size_t) floor(std::pow((double) np, (double) 1.0/dim) + 0.5);
	CoordVector<dim, std::size_t> procCoords;
	for (std::size_t d = 0; d < dim - 1; ++d)
	{
		procCoords[d] = procInd % dimProcs;
		procInd = (procInd - procCoords[d]) / dimProcs;
	}
	procCoords[dim-1] = procInd;

	// iterate all possible boundary sub-domains (corners, edges, faces)
	CoordVector<dim, int> bndDom (-1);
	std::size_t nBndDom = std::pow(3, dim);
	for (std::size_t i = 0; i < nBndDom; ++i)
	{
		// find 0s (i.e.: in which dimensions our indices have to be iterated over)
		// find 1s (i.e.: possible slave dimensions)
		// find -1s (i.e.: possible master dimensions)
		std::vector<std::size_t> zeros;
		std::vector<std::size_t> ones;
		std::vector<std::size_t> negones;

		for (std::size_t d = 0; d < dim; ++d)
		{
			if (bndDom[d] == 0)
				zeros.push_back(d);
			else if (bndDom[d] == 1)
			{
				// only possible slaves in this direction if proc is not at bnd in this direction
				if (procCoords[d] < dimProcs-1)
					ones.push_back(d);
			}
			else
			{
				// only possible masters in this direction if proc is not at bnd in this direction
				if (procCoords[d] > 0)
					negones.push_back(d);
			}
		}

		// if there are elements in negones, we are slave
		if (negones.size())
		{
			// calculate master proc coords in proc grid
			CoordVector<dim, std::size_t> masterCoords = procCoords;
			for (std::size_t j = 0; j < negones.size(); ++j)
				masterCoords[negones[j]] -= 1;

			// calculate master proc index from that
			std::size_t mInd = masterCoords[0];
			std::size_t nDims = 1;
			for (std::size_t d = 1; d < dim; ++d)
			{
				nDims *= dimProcs;
				mInd += masterCoords[d] * nDims;
			}

			// only work if there are indices in this bnd sub-dom
			std::size_t nInds = 1;
			for (std::size_t j = 0; j < zeros.size(); ++j)
				nInds *= m_vNumVertexPerDim[zeros[j]] - 2;

			if (nInds)
			{
				// collect all indices in this boundary sub-domain
				std::vector<std::size_t> inds(nInds);
				CoordVector<dim, std::size_t> indexCoords(1);
				for (std::size_t d = 0; d < dim; ++d)
				{
					if (bndDom[d] == -1)
						indexCoords[d] = 0;
					else if (bndDom[d] == 1)
						indexCoords[d] = m_vNumVertexPerDim[d]-1;
				}

				for (std::size_t j = 0; j < nInds; ++j)
				{
					inds[j] = vertex(indexCoords);

					// next coords
					std::size_t k = 0;
					while (k < zeros.size())
					{
						indexCoords[zeros[k]] += 1;
						if (indexCoords[zeros[k]] == m_vNumVertexPerDim[zeros[k]]-1)
						{
							indexCoords[zeros[k]] = 1;
							++k;
						}
						else
							break;
					}
				}

				// create slave interface for all indices of this boundary sub-domain
				// or add to existing one
				slaveLayout.add_interface(mInd, inds);
			}
		}

		// if there are elements in ones, then we are master
		else if (ones.size())
		{
			// only work if there are indices in this bnd sub-dom
			std::size_t nInds = 1;
			for (std::size_t j = 0; j < zeros.size(); ++j)
				nInds *= m_vNumVertexPerDim[zeros[j]] - 2;

			if (nInds)
			{
				// collect all indices in this boundary sub-domain
				std::vector<std::size_t> inds(nInds);
				CoordVector<dim, std::size_t> indexCoords(1);
				for (std::size_t d = 0; d < dim; ++d)
				{
					if (bndDom[d] == -1)
						indexCoords[d] = 0;
					else if (bndDom[d] == 1)
						indexCoords[d] = m_vNumVertexPerDim[d]-1;
				}

				for (std::size_t j = 0; j < nInds; ++j)
				{
					inds[j] = vertex(indexCoords);

					// next coords
					std::size_t k = 0;
					while (k < zeros.size())
					{
						indexCoords[zeros[k]] += 1;
						if (indexCoords[zeros[k]] == m_vNumVertexPerDim[zeros[k]]-1)
						{
							indexCoords[zeros[k]] = 1;
							++k;
						}
						else
							break;
					}
				}

				// loop neighboring procs
				std::size_t nSlaves = std::pow(2, ones.size()) - 1;
				CoordVector<dim, std::size_t> slaveCoords = procCoords;
				for (std::size_t j = 0; j < nSlaves; ++j)
				{
					// next slave coords
					std::size_t k = 0;
					while (k < ones.size())
					{
						slaveCoords[ones[k]] += 1;
						if (slaveCoords[ones[k]] == procCoords[ones[k]] + 2)
						{
							slaveCoords[ones[k]] = procCoords[ones[k]];
							++k;
						}
						else
							break;
					}

					// get slave proc ID from slave coords
					std::size_t sInd = slaveCoords[0];
					std::size_t nDims = 1;
					for (std::size_t d = 1; d < dim; ++d)
					{
						nDims *= dimProcs;
						sInd += slaveCoords[d] * nDims;
					}

					// add indices to master interface for current slave
					masterLayout.add_interface(sInd, inds);
				}
			}
		}

		// increase bndDom id
		std::size_t d = 0;
		while (d < dim)
		{
			bndDom[d] += 1;
			if (bndDom[d] == 2)
			{
				bndDom[d] = -1;
				++d;
			}
			else
				break;
		}
	}


	// reorder all interfaces (to make sure the correct slave/master pairs are mapped together)
	auto it = masterLayout.begin();
	auto itEnd = masterLayout.end();
	for (; it != itEnd; ++it)
	{
		auto& interface = it->second;
		std::sort(interface.begin(), interface.end());
	}

	it = slaveLayout.begin();
	itEnd = slaveLayout.end();
	for (; it != itEnd; ++it)
	{
		auto& interface = it->second;
		std::sort(interface.begin(), interface.end());
	}
}


// explicit template declarations
template class StructuredGrid<1>;
template class StructuredGrid<2>;
template class StructuredGrid<3>;
