/*
 * ell_matrix.h
 *
 * author: a.vogel@rub.de
 *
 */

#ifndef ELL_MATRIX_CUDA_H
#define ELL_MATRIX_CUDA_H

#include <vector>
#include <ostream>
#include <type_traits>
#include <cassert>
#include <iostream>
#include <stdexcept>

#include "lil_matrix.h"
#include "vector.h"

#include "parallel/layout.h"
#include "parallel/parallel_storage_type.h"

#include "tdo/tdo.h"

#include <cuda.h>
#include <cuda_runtime.h>

#define CUDA_CALL( call )                                                                                          \
    {                                                                                                                  \
    cudaError_t err = call;                                                                                          \
    if ( cudaSuccess != err)                                                                                         \
        fprintf(stderr, "CUDA error for %s in %d of %s : %s.\n", #call , __LINE__ , __FILE__ ,cudaGetErrorString(err));\
    }

__global__ 
void printInitialResult_GPU(double* res0, double* m_minRes, double* m_minRed)
{
	printf("    0    %e    %9.3e      -----        --------      %9.3e    \n", *res0, *m_minRes, *m_minRed);
}


__global__ 
void printResult_GPU(size_t* step, double* res, double* m_minRes, double* lastRes, double* res0, double* m_minRed)
{
	if(*step < 10)
	printf("    %d    %e    %9.3e    %9.3e    %e    %9.3e    \n", *step, *res, *m_minRes, (*res)/(*lastRes), (*res)/(*res0), *m_minRed);

	else
	printf("   %d    %e    %9.3e    %9.3e    %e    %9.3e    \n", *step, *res, *m_minRes, (*res)/(*lastRes), (*res)/(*res0), *m_minRed);
}



/// r = A*x
__global__ 
void Apply_GPU(	
	const std::size_t num_rows, 
	const std::size_t num_cols_per_row,
	const double* value,
	const std::size_t* index,
	const double* x,
	double* r)
{
	int id = blockDim.x * blockIdx.x + threadIdx.x;
	
	if ( id < num_rows )
	{
		double dot = 0;

		for ( int n = 0; n < num_cols_per_row; n++ )
		{
			int col = index [ num_cols_per_row * id + n ];
			double val = value [ num_cols_per_row * id + n ];
			dot += val * x [ col ];
		}
		r[id] = dot;
	}
	
}

/// r = A^T * x
/// NOTE: This kernel should be run with A's number of rows as the number of threads
/// e.g., r's size = 9, A's size = 25 x 9, x's size = 25
/// ApplyTransposed_GPU<<<1, 25>>>()
__global__ 
void ApplyTransposed_GPU(	
	const std::size_t num_rows, 
	const std::size_t num_cols_per_row,
	const double* value,
	const std::size_t* index,
	const double* x,
	double* r)
{
	int id = blockDim.x * blockIdx.x + threadIdx.x;

	if ( id < num_rows )
	{
		// r[id] = 0;
		// __syncthreads();

		for ( int n = 0; n < num_cols_per_row; n++ )
		{
			int col = index [ num_cols_per_row * id + n ];
			float val = value [ num_cols_per_row * id + n ];
			atomicAdd_double( &r[col], val*x[id] );
		}
	}
}

/// r = r - A*x
__global__ 
void UpdateResiduum_GPU(
	const std::size_t num_rows, 
	const std::size_t num_cols_per_row,
	const double* value,
	const std::size_t* index,
	const double* x,
	double* r)
{
  	int id = blockDim.x * blockIdx.x + threadIdx.x;

	if ( id < num_rows )
	{
		double dot = 0.0;

		for ( int n = 0; n < num_cols_per_row; n++ )
		{
			std::size_t col = index [ num_cols_per_row * id + n ];
			double val = value [ num_cols_per_row * id + n ];
			dot += val * x [ col ];
		}
		r[id] = r[id] - dot;
	}
	
}

/// r = b - A*x
__global__ 
void ComputeResiduum_GPU(	
	const std::size_t num_rows, 
	const std::size_t num_cols_per_row,
	const double* value,
	const std::size_t* index,
	const double* x,
	double* r,
	double* b)
{
	int id = blockDim.x * blockIdx.x + threadIdx.x;

	if ( id < num_rows )
	{
		double dot = 0.0;

		for ( int n = 0; n < num_cols_per_row; n++ )
		{
			int col = index [ num_cols_per_row * id + n ];
			double val = value [ num_cols_per_row * id + n ];
			dot += val * x [ col ];
		}
		r[id] = b[id] - dot;
	}
	
}


// p = z + p * beta;
__global__ 
void calculateDirectionVector(	
	size_t* d_step,
	double* d_p, 
	double* d_z, 
	double* d_rho, 
	double* d_rho_old,
	size_t num_rows)
{
	int id = blockDim.x * blockIdx.x + threadIdx.x;

	if ( id < num_rows )
	{
		// if(step == 1) p = z;
		if(*d_step == 1)
		{ 
			d_p[id] = d_z[id]; 
		}
		
		else
		{
			// p *= (rho / rho_old)
			d_p[id] = d_p[id] * ( *d_rho / (*d_rho_old) );

			// __syncthreads();
		
			// p += z;
			d_p[id] = d_p[id] + d_z[id];
		}
	}
}


// alpha = rho / (p * z); 
__global__ 
void calculateAlpha(
	double* d_alpha, 
	double* d_rho, 
	double* d_p, 
	double* d_z, 
	double* alpha_temp,
	size_t num_rows)
{
	int id = blockDim.x * blockIdx.x + threadIdx.x;

	if ( id == 0 )
		*alpha_temp = 0.0;
	__syncthreads();

	if ( id < num_rows )
	{
		// ( p * z )
		atomicAdd_double( alpha_temp, d_p[id] * d_z[id] );
	}

	__syncthreads();

	if ( id == 0 )
		*d_alpha = *d_rho / (*alpha_temp);
}

__host__
void calculateAlpha_test(
	double* d_alpha, 
	double* d_rho, 
	double* d_p, 
	double* d_z, 
	double* d_alpha_temp,
	size_t num_rows,
	dim3 gridDim,
	dim3 blockDim)
{

	setToZero<<<1,1>>>( d_alpha_temp, 1);
	// cudaDeviceSynchronize();

	// alpha_temp = () p * z )
	// dotProduct<<<gridDim,blockDim>>>(d_alpha_temp, d_p, d_z, num_rows);
	dotProduct_test(d_alpha_temp, d_p, d_z, num_rows, gridDim, blockDim);
	// cudaDeviceSynchronize();

	// d_alpha = *d_rho / (*alpha_temp)
	divide_GPU<<<1,1>>>(d_alpha, d_rho, d_alpha_temp);
	// cudaDeviceSynchronize();

}
template <class T>
class ELLMatrix
{
	public:
		/// entry type
		typedef T value_type;

		/// size type
		typedef std::size_t size_type;

	public:
		ELLMatrix(std::shared_ptr<AlgebraLayout> spLayouts = nullptr)
			: m_storageMask(PST_UNDEFINED), m_spAlgebraLayout(spLayouts)
		{
			m_max_row_size = m_num_rows = m_num_cols = 0;
		};

		/// constructor 
		ELLMatrix(const LILMatrix<T>& mat, std::shared_ptr<AlgebraLayout> spLayouts = nullptr)
			: m_storageMask(PST_UNDEFINED), m_spAlgebraLayout(spLayouts)
		{

			m_max_row_size = mat.max_row_size();
			m_num_rows = mat.num_rows();
			m_num_cols = mat.num_cols();

			m_vValue.resize(m_num_rows * m_max_row_size, 0.0);
			m_vIndex.resize(m_num_rows * m_max_row_size, 0);

			// copy rows
			for(size_type r = 0; r < m_num_rows; ++r){
				int k = 0;
				auto itEnd = mat.end(r);
				for(auto it = mat.begin(r); it != itEnd; ++it, ++k){
					m_vValue[r * m_max_row_size + k] = it.value();
					m_vIndex[r * m_max_row_size + k] = it.index();
				}
			}
		}

















		// TDO constructor
		ELLMatrix(size_t N, double rho)
		{
			std::cout << "TDO stiffness matrix\n";
			size_type dim = 2; //TODO:

			  // calculate the number of elements in the domain                                                               
			size_t numElements = pow(N,dim);
			size_t numNodesPerDim = N + 1;
			size_t numNodes = numNodesPerDim*numNodesPerDim;
			m_num_rows = numNodes*dim;
			m_num_cols = numNodes*dim;

			// calculate h
			float h = 1.0/N;

			// create an array of nodes
			// std::vector<Node> node;
			
			for ( int i = 0 ; i < numNodes ; ++i )
				node.push_back(Node(i));


			calculateNodeCoordinates(&node[0], numNodes, numNodesPerDim, h);
				
				
			// creating an array of elements
				// std::vector<Element> element;
				
				for ( int i = 0 ; i < numElements ; i++ )
				element.push_back( Element(i) );
				
				
			// adding node indices
			for ( int i = 0 ; i < numElements ; i++ )
			{
				element[i].addNode(&node[ i + i/N ]);   // lower left node
				element[i].addNode(&node[ i + i/N + 1]);   // lower right node
				element[i].addNode(&node[ i + i/N + N + 1]);   // upper left node
				element[i].addNode(&node[ i + i/N + N + 2]);   // upper right node
			}
			

			// flattened global matrix
			std::vector<double> K = {	4,	1,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0, \
										1,	4,	1,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0, \
										0,	1,	8,	2,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0, \
										0,	0,	2,	8,	1,	0,	1,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0, \
										0,	0,	0,	1,	4,	1,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0, \
										0,	0,	0,	0,	1,	4,	0,	0,	1,	0,	0,	0,	0,	0,	0,	0,	0,	0, \
										0,	0,	0,	1,	0,	0,	8,	2,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0, \
										0,	0,	0,	0,	0,	0,	2,	8,	2,	0,	0,	0,	0,	0,	0,	0,	0,	0, \
										0,	0,	0,	0,	0,	1,	0,	2,	16,	4,	0,	0,	0,	0,	0,	0,	0,	0, \
										0,	0,	0,	0,	0,	0,	0,	0,	4,	16,	2,	0,	1,	0,	0,	0,	0,	0, \
										0,	0,	0,	0,	0,	0,	0,	0,	0,	2,	8,	2,	0,	0,	0,	0,	0,	0, \
										0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	2,	8,	0,	0,	1,	0,	0,	0, \
										0,	0,	0,	0,	0,	0,	0,	0,	0,	1,	0,	0,	4,	1,	0,	0,	0,	0, \
										0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	1,	4,	1,	0,	0,	0, \
										0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	1,	0,	1,	8,	2,	0,	0, \
										0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	2,	8,	1,	0, \
										0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	1,	4,	1, \
										0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	1,	4  };
						

			// CUDA

			CUDA_CALL( cudaMalloc( (void**)&d_K, sizeof(double) * 18 * 18 )     );
			CUDA_CALL( cudaMalloc( (void**)&d_max_row_size, sizeof(size_t) )    );
			CUDA_CALL( cudaMalloc( (void**)&d_mutex, sizeof(int) ) );

			CUDA_CALL( cudaMemset(d_max_row_size, 0, sizeof(size_t)) );
			CUDA_CALL( cudaMemset(d_mutex, 0, sizeof(int)) );
			
			CUDA_CALL( cudaMemcpy(d_K, &K[0], sizeof(double) * 18 * 18 , cudaMemcpyHostToDevice) ); 


			// calculate global matrix's max_row_size
			getMaxRowSize<<< 1 , 18 >>>(d_K, d_max_row_size, d_mutex, 18);
			CUDA_CALL( cudaMemcpy(&m_max_row_size, d_max_row_size, sizeof(size_t), cudaMemcpyDeviceToHost ) ); 
			// std::cout << m_max_row_size << std::endl;
			
			// allocate device memory for global stiffness matrix's ELLPACK value and index vectors
			CUDA_CALL( cudaMalloc( (void**)&d_K_value, sizeof(double) * 18 * m_max_row_size )     );
			CUDA_CALL( cudaMalloc( (void**)&d_K_index, sizeof(size_t) * 18 * m_max_row_size )     );
			
			// transform K to ELLPACK
			transformToELL_GPU<<<1, 18>>>(d_K, d_K_value, d_K_index, m_max_row_size, 18);
			
			
			// deallocate big K matrix, no needed now
			cudaFree( d_K );
			
			
			// copy and allocate the node index of each element
			
			std::vector<size_t*> d_node_index(numElements);
			
			for ( int i = 0 ; i < numElements ; i++ )
			{
				CUDA_CALL( cudaMalloc( (void**)&d_node_index[i], sizeof(size_t) * 4 )     );
				CUDA_CALL( cudaMemcpy( d_node_index[i], element[i].getNodeGlobalIndex() , sizeof(size_t) * 4 , cudaMemcpyHostToDevice ) ); 
			}


			// obtain k elements' value and index vectors
			// allocate k element stiffness matrices

			d_ke_value.resize(numElements);
			d_ke_index.resize(numElements);


			// allocate and copy elements' ELLPACK stiffness matrices to device (value and index vectors)
			for ( int i = 0 ; i < numElements ; i++ )
			{
				CUDA_CALL( cudaMalloc( (void**)&d_ke_value[i], sizeof(double) * 24 )     );
				CUDA_CALL( cudaMalloc( (void**)&d_ke_index[i], sizeof(size_t) * 24 )     );

				CUDA_CALL( cudaMemcpy( d_ke_value[i], element[i].getValueAddress() , sizeof(double) * 24 , cudaMemcpyHostToDevice ) ); 
				CUDA_CALL( cudaMemcpy( d_ke_index[i], element[i].getIndexAddress() , sizeof(size_t) * 24 , cudaMemcpyHostToDevice ) ); 
			}
			


			// array of the initial design variable

			std::vector<double> design(numElements);

			for ( int i = 0 ; i < numElements ; i++ )
				design.push_back(rho);

			double* d_design = nullptr;
				
			CUDA_CALL( cudaMalloc( (void**)&d_design, sizeof(double) * numElements )     );
			CUDA_CALL( cudaMemcpy( d_design, &design[0] , sizeof(double) * numElements , cudaMemcpyHostToDevice ) ); 
			


			// allocate and copy the empty global matrix

			Element global;
			
			double* d_KG_value;
			size_type* d_KG_index;
			
			CUDA_CALL( cudaMalloc( (void**)&d_KG_value, sizeof(double) * 72 )     );
			CUDA_CALL( cudaMalloc( (void**)&d_KG_index, sizeof(size_t) * 72 )     );
			CUDA_CALL( cudaMemcpy( d_KG_value, global.getValueAddress() , sizeof(double) * 72 , cudaMemcpyHostToDevice ) ); 
			CUDA_CALL( cudaMemcpy( d_KG_index, global.getIndexAddress() , sizeof(size_t) * 72 , cudaMemcpyHostToDevice ) ); 
			

			// add local stiffness matrices into the global

			// for ( int i = 0 ; i < numElements ; i++ )
			// {
			//     assembleGrid_GPU<<<1, 8>>>( 2, 2, d_ke_value[i], d_ke_index[i], element[i].max_row_size(), element[i].num_rows(), d_KG_value, d_K_index, global.max_row_size(), global.num_rows(), d_node_index[i] );
			//     cudaDeviceSynchronize();
			// }

			dim3 blockDim(8,8,1);
			//     assembleGrid2D_GPU<<<1, blockDim>>>( 2, 2, d_ke_value[0], d_ke_index[0], element[0].max_row_size(), element[0].num_rows(), d_KG_value, d_K_index, global.max_row_size(), global.num_rows(), d_node_index[0] );

			for ( int i = 0 ; i < numElements ; i++ )
				{
					assembleGrid2D_GPU<<<1, blockDim>>>( 2, 2, d_ke_value[i], d_ke_index[i], element[i].max_row_size(), element[i].num_rows(), d_KG_value, d_K_index, global.max_row_size(), global.num_rows(), d_node_index[i] );
					cudaDeviceSynchronize();
				}


			// printVector_GPU<<<1,72>>> ( d_KG_value, 72 );
			// // printVector_GPU<<<1,72>>> ( d_K_index, 72 );
			// cudaDeviceSynchronize();
		}


		/// CUDA addresses ///

		/// Returns addresses of device value and index vectors
		double* getDeviceValueAddress() { return d_K_value; }
		std::size_t* getDeviceIndexAddress() { return d_K_index; }
















        
		/// Returns addresses of value and index vectors
		const T* getValueAddress() const { return &m_vValue[0]; }
		const std::size_t* getIndexAddress() const { return &m_vIndex[0]; }

		/// return number of rows
		size_type num_rows() const {return m_num_rows;}

		/// return number of columns
		size_type num_cols() const {return m_num_cols;}

		/// const entry access
		T operator()(size_type r, size_type c) const
		{
			assert(r < num_rows() && c < num_cols() && "Invalid index requested.");

			for(size_type k = 0; k < m_max_row_size; ++k){
				if(m_vIndex[r * m_max_row_size + k] == c)
					return m_vValue[r * m_max_row_size + k];
			}
			return 0.0;
		}

		/// number of stored nonzeros
		size_type nnz() const {return m_vValue.size();}

		// max row size
		size_type max_row_size() const {return m_max_row_size;}

	// REVIEW: #########################################################################################################################
	public:
		template <class ValueType, class IndexType>
		class RowIterator 
		{
			public:
				typedef std::bidirectional_iterator_tag  iterator_category;
				typedef typename std::remove_const<ValueType>::type	value_type; // mutable value type required for std::iterator_traits
				typedef std::ptrdiff_t	difference_type;
				typedef ValueType*		pointer;		// possibly const value type required for std::iterator_traits
				typedef ValueType&		reference;		// possibly const value type required for std::iterator_traits
				typedef typename std::remove_const<IndexType>::type	index_type;	


			public:
				RowIterator() : p(0), ind(0), j(0) {}
				RowIterator(ValueType* _p, IndexType* _ind, size_type _j) : 	p(_p), ind(_ind), j(_j) {}
				RowIterator(const RowIterator<value_type,index_type>& o) : p(o.p), ind(o.ind), j(o.j) {}

				bool operator!=(const RowIterator<const value_type, const index_type>& o) const {return j != o.j || p != o.p;}
				bool operator==(const RowIterator<const value_type, const index_type>& o) const {return j == o.j && p == o.p;}

				RowIterator& operator++() {++j; return *this;}
				RowIterator& operator++(int) {RowIterator tmp(*this); ++(*this); return tmp;}

				RowIterator& operator--() {--j; return *this;}
				RowIterator& operator--(int) {RowIterator tmp(*this); --(*this); return tmp;}

				ValueType& value() const {return p[j];}
				ValueType& operator*() const {return value();}
				ValueType* operator->() const {return &value();}

				size_type index() const {return ind[j];}

			protected:
				ValueType* p;
				IndexType* ind;
				size_type j;
		};

		typedef RowIterator<T, size_type> row_iterator;
		typedef RowIterator<const T, const size_type> const_row_iterator;

		row_iterator begin(size_type row)		{return row_iterator(&m_vValue[row * m_max_row_size], &m_vIndex[row * m_max_row_size], 0);}
		row_iterator end(size_type row)			{return row_iterator(&m_vValue[row * m_max_row_size], &m_vIndex[row * m_max_row_size], m_max_row_size);}

		const_row_iterator begin(size_type row) const {return const_row_iterator(&m_vValue[row * m_max_row_size], &m_vIndex[row * m_max_row_size], 0);}
		const_row_iterator end(size_type row) 	const {return const_row_iterator(&m_vValue[row * m_max_row_size], &m_vIndex[row * m_max_row_size], m_max_row_size);}


	public:
		/// set the storage type
		void set_storage_type(uint8_t type)
		{
			if(type == PST_UNIQUE) type = PST_UNIQUE | PST_ADDITIVE;
			m_storageMask = type;
		}

		/// return if storage type is contained in the actual storage type
		bool has_storage_type(uint8_t type) const
		{
			return type == PST_UNDEFINED ? m_storageMask == PST_UNDEFINED : (m_storageMask & type) == type;
		}

		/// return the actual combination of storages types
		uint8_t get_storage_mask() const
		{
			return m_storageMask;
		}

		/// return layout
		std::shared_ptr<AlgebraLayout> layouts() const {return m_spAlgebraLayout;}

		/// set layouts
		void set_layouts(std::shared_ptr<AlgebraLayout> spLayouts) {m_spAlgebraLayout = spLayouts;}

	protected:
		/// storage type
		uint8_t m_storageMask;

		/// algebra layouts
		std::shared_ptr<AlgebraLayout> m_spAlgebraLayout;		

	protected:
		// data storage
		size_type m_num_rows, m_num_cols, m_max_row_size;

		// Ellpack storage for entries and indices 
		std::vector<T> m_vValue;
		std::vector<size_type> m_vIndex;


		// TDO variables
		std::vector<Node> node;
		std::vector<Element> element;
		
		std::vector<double*> d_ke_value;
		std::vector<size_t*> d_ke_index;

		// CUDA 
		// device pointers
		double* d_K             = nullptr;
		double* d_K_value       = nullptr;
		size_t* d_K_index       = nullptr;
		size_t* d_max_row_size  = nullptr;
		int* d_mutex            = nullptr;
				
};

/// output for dense matrices
// template <typename T>
// std::ostream& operator<<(std::ostream& stream, const ELLMatrix<T>& m)
// {

// 	if (!m.num_cols()) return stream << "[]" << std::endl;

// 	for (std::size_t i = 0; i < m.num_rows(); ++i){
// 		stream << "[";
// 		auto iterEnd = m.end(i);		
// 		for (auto iter = m.begin(i); iter != iterEnd; ++iter){
// 			stream << "(" << iter.value() << "," << iter.index() << ")";
// 		}
// 		stream << "]";
// 		stream << std::endl;
// 	}

// 	return stream;
// }


////////////////////////////////////////
// some help methods
////////////////////////////////////////


/// r = A*x
inline void Apply(Vector<double>& r,  const ELLMatrix<double>& A, const Vector<double>& x){


	// if(!x.has_storage_type(PST_CONSISTENT))
	// 	throw(std::runtime_error("Solution expected to be consistent"));

	if(x.size() != A.num_cols()){
		throw(std::runtime_error("Matrix columns must match Vector size."));
	}

	r.resize(A.num_rows());

	for(size_t i = 0; i < A.num_cols(); ++i){
		r[i] = 0.0;
	}


	// [START] CUDA implementation section -----------------------------------------------------------------------------------------------
	// Host pointers

    const auto value = A.getValueAddress();
    const auto index = A.getIndexAddress();

    // Device pointers

    double* d_value = NULL;
    std::size_t* d_index = NULL;
    double* d_x = NULL;
    double* d_r = NULL;

    // Allocate memory on device

    CUDA_CALL( cudaMalloc( (void**)&d_value, A.max_row_size() * A.num_rows() * sizeof(double) ) ); // A.num_rows() * A.num_cols()
    CUDA_CALL( cudaMalloc( (void**)&d_index, A.max_row_size() * A.num_rows() * sizeof(std::size_t) ) );
    CUDA_CALL( cudaMalloc( (void**)&d_x, x.size() * sizeof(double) ) );
    CUDA_CALL( cudaMalloc( (void**)&d_r, r.size() * sizeof(double) ) );

    CUDA_CALL( cudaMemcpy( d_value, value, A.max_row_size() * A.num_rows() * sizeof(double), cudaMemcpyHostToDevice) );
    CUDA_CALL( cudaMemcpy( d_index, index, A.max_row_size() * A.num_rows() * sizeof(std::size_t), cudaMemcpyHostToDevice) );
    CUDA_CALL( cudaMemcpy( d_x, &x[0], x.size() * sizeof(double), cudaMemcpyHostToDevice) );
    CUDA_CALL( cudaMemcpy( d_r, &r[0], r.size() * sizeof(double), cudaMemcpyHostToDevice) );

    Apply_GPU<<<1,A.num_rows()>>>( A.num_rows(), A.max_row_size(), d_value, d_index, d_x, d_r );

	// This part below is temporary. The final program will have r remain in the device.

    CUDA_CALL( cudaMemcpy( &r[0], d_r, r.size() * sizeof(double), cudaMemcpyDeviceToHost) );

    CUDA_CALL( cudaFree(d_value) );
    CUDA_CALL( cudaFree(d_index) );
    CUDA_CALL( cudaFree(d_x) );
    CUDA_CALL( cudaFree(d_r) );

	cudaDeviceSynchronize();
	// [END] CUDA implementation section -----------------------------------------------------------------------------------------------

	r.set_storage_type(PST_ADDITIVE);

}

/// r = A^T * x
inline void ApplyTransposed(Vector<double>& r,  const ELLMatrix<double>& A, const Vector<double>& x){

	// printf("ApplyTransposed()-CUDA\n");

	if(x.size() != A.num_rows()){
		throw(std::runtime_error("Matrix columns must match Vector size."));
	}

	if(r.size() != A.num_cols()){
		throw(std::runtime_error("Matrix columns must match Vector size."));
	}

	for(size_t i = 0; i < A.num_cols(); ++i){
		r[i] = 0.0;
	}

	// [START] CUDA implementation section -----------------------------------------------------------------------------------------------
	// Host pointers

    const auto value = A.getValueAddress();
    const auto index = A.getIndexAddress();

    // Device pointers

    double* d_value = NULL;
    std::size_t* d_index = NULL;
    double* d_x = NULL;
    double* d_r = NULL;


    // Allocate memory on device

    CUDA_CALL( cudaMalloc( (void**)&d_value, A.max_row_size() * A.num_rows() * sizeof(double) ) ); // A.num_rows() * A.num_cols()
    CUDA_CALL( cudaMalloc( (void**)&d_index, A.max_row_size() * A.num_rows() * sizeof(std::size_t) ) );
    CUDA_CALL( cudaMalloc( (void**)&d_x, x.size() * sizeof(double) ) );
    CUDA_CALL( cudaMalloc( (void**)&d_r, r.size() * sizeof(double) ) );

    CUDA_CALL( cudaMemcpy( d_value, value, A.max_row_size() * A.num_rows() * sizeof(double), cudaMemcpyHostToDevice) );
    CUDA_CALL( cudaMemcpy( d_index, index, A.max_row_size() * A.num_rows() * sizeof(std::size_t), cudaMemcpyHostToDevice) );
    CUDA_CALL( cudaMemcpy( d_x, &x[0], x.size() * sizeof(double), cudaMemcpyHostToDevice) );
    CUDA_CALL( cudaMemcpy( d_r, &r[0], r.size() * sizeof(double), cudaMemcpyHostToDevice) );

    ApplyTransposed_GPU<<<1,A.num_rows()>>>( A.num_rows(), A.max_row_size(), d_value, d_index, d_x, d_r );

	// This part below is temporary. The final program will have r remain in the device.

    CUDA_CALL( cudaMemcpy( &r[0], d_r, r.size() * sizeof(double), cudaMemcpyDeviceToHost) );

    CUDA_CALL( cudaFree(d_value) );
    CUDA_CALL( cudaFree(d_index) );
    CUDA_CALL( cudaFree(d_x) );
    CUDA_CALL( cudaFree(d_r) );

	// [END] CUDA implementation section -----------------------------------------------------------------------------------------------


	r.set_storage_type(PST_ADDITIVE);
}


/// r = r - A*x
inline void UpdateResiduum(Vector<double>& r,  const ELLMatrix<double>& A, const Vector<double>& x){

	// std::cout << "UpdateResiduum(): c[8] = " << x[8] << std::endl;
	
	// if(!r.has_storage_type(PST_ADDITIVE))
	// 	throw(std::runtime_error("Residuum expected to be additive"));

	// if(!x.has_storage_type(PST_CONSISTENT))
	// 	throw(std::runtime_error("solution expected to be consistent"));

		// std::cout << "UpdateResiduum: x[8] = " << x[8] << std::endl;


	if(x.size() != A.num_cols())
		throw(std::runtime_error("Matrix columns must match Vector size."));

	r.resize(A.num_rows());


// [START] CUDA implementation section -----------------------------------------------------------------------------------------------
	// Host pointers

    const auto value = A.getValueAddress();
    const auto index = A.getIndexAddress();

    // Device pointers

    double* d_value = NULL;
    std::size_t* d_index = NULL;
    double* d_x = NULL;
    double* d_r = NULL;

    // Allocate memory on device

    CUDA_CALL( cudaMalloc( (void**)&d_value, A.max_row_size() * A.num_rows() * sizeof(double) ) ); // A.num_rows() * A.num_cols()
    CUDA_CALL( cudaMalloc( (void**)&d_index, A.max_row_size() * A.num_rows() * sizeof(std::size_t) ) );
    CUDA_CALL( cudaMalloc( (void**)&d_x, x.size() * sizeof(double) ) );
    CUDA_CALL( cudaMalloc( (void**)&d_r, r.size() * sizeof(double) ) );

    CUDA_CALL( cudaMemcpy( d_value, value, A.max_row_size() * A.num_rows() * sizeof(double), cudaMemcpyHostToDevice) );
    CUDA_CALL( cudaMemcpy( d_index, index, A.max_row_size() * A.num_rows() * sizeof(std::size_t), cudaMemcpyHostToDevice) );
    CUDA_CALL( cudaMemcpy( d_x, &x[0], x.size() * sizeof(double), cudaMemcpyHostToDevice) );
    CUDA_CALL( cudaMemcpy( d_r, &r[0], r.size() * sizeof(double), cudaMemcpyHostToDevice) );


    UpdateResiduum_GPU<<<1,A.num_rows()>>>( A.num_rows(), A.max_row_size(), d_value, d_index, d_x, d_r );
	cudaDeviceSynchronize();

	// This part below is temporary. The final program will have r remain in the device.

    CUDA_CALL( cudaMemcpy( &r[0], d_r, r.size() * sizeof(double), cudaMemcpyDeviceToHost) );

    CUDA_CALL( cudaFree(d_value) );
    CUDA_CALL( cudaFree(d_index) );
    CUDA_CALL( cudaFree(d_x) );
    CUDA_CALL( cudaFree(d_r) );


	// [END] CUDA implementation section -----------------------------------------------------------------------------------------------

	r.set_storage_type(PST_ADDITIVE);
}



/// r = b - A*x
inline void ComputeResiduum(Vector<double>& r,  const Vector<double>& b, const ELLMatrix<double>& A, const Vector<double>& x){


	// std::cout << "ComputeResiduum: c[8] = " << x[8] << std::endl;

	// if(!b.has_storage_type(PST_ADDITIVE))
	// 	throw(std::runtime_error("Residuum expected to be additive"));

	// if(!x.has_storage_type(PST_CONSISTENT))
	// 	throw(std::runtime_error("Solution expected to be consistent"));

	if(x.size() != A.num_cols()){
		throw(std::runtime_error("Matrix columns must match Vector size."));
	}

	r.resize(A.num_rows());

// [START] CUDA implementation section -----------------------------------------------------------------------------------------------
	// Host pointers

    const auto value = A.getValueAddress();
    const auto index = A.getIndexAddress();

    // Device pointers

    double* d_value = NULL;
    std::size_t* d_index = NULL;
    double* d_x = NULL;
    double* d_r = NULL;
	double* d_b = NULL;

    // Allocate memory on device

    CUDA_CALL( cudaMalloc( (void**)&d_value, A.max_row_size() * A.num_rows() * sizeof(double) ) ); // A.num_rows() * A.num_cols()
    CUDA_CALL( cudaMalloc( (void**)&d_index, A.max_row_size() * A.num_rows() * sizeof(std::size_t) ) );
    CUDA_CALL( cudaMalloc( (void**)&d_x, x.size() * sizeof(double) ) );
    CUDA_CALL( cudaMalloc( (void**)&d_r, r.size() * sizeof(double) ) );
    CUDA_CALL( cudaMalloc( (void**)&d_b, b.size() * sizeof(double) ) );

    CUDA_CALL( cudaMemcpy( d_value, value, A.max_row_size() * A.num_rows() * sizeof(double), cudaMemcpyHostToDevice) );
    CUDA_CALL( cudaMemcpy( d_index, index, A.max_row_size() * A.num_rows() * sizeof(std::size_t), cudaMemcpyHostToDevice) );
    CUDA_CALL( cudaMemcpy( d_x, &x[0], x.size() * sizeof(double), cudaMemcpyHostToDevice) );
    CUDA_CALL( cudaMemcpy( d_r, &r[0], r.size() * sizeof(double), cudaMemcpyHostToDevice) );
    CUDA_CALL( cudaMemcpy( d_b, &b[0], b.size() * sizeof(double), cudaMemcpyHostToDevice) );

	
    ComputeResiduum_GPU<<<1,A.num_rows()>>>( A.num_rows(), A.max_row_size(), d_value, d_index, d_x, d_r, d_b );
	cudaDeviceSynchronize();

    CUDA_CALL( cudaMemcpy( &r[0], d_r, r.size() * sizeof(double), cudaMemcpyDeviceToHost) );

	// This part below is temporary. The final program will have r remain in the device.
    CUDA_CALL( cudaFree(d_value) );
    CUDA_CALL( cudaFree(d_index) );
    CUDA_CALL( cudaFree(d_x) );
    CUDA_CALL( cudaFree(d_r) );

	// [END] CUDA implementation section -----------------------------------------------------------------------------------------------

	r.set_storage_type(PST_ADDITIVE);
}


/// computes P^T * A * P
inline void MultiplyPTAP(ELLMatrix<double>& PTAP,  const ELLMatrix<double>& A, const ELLMatrix<double>& P){

	if( A.num_cols() != P.num_rows() || P.num_rows() != A.num_rows() ){
		std::cout << "A: " << A.num_rows() << " x " << A.num_cols() << std::endl;
		std::cout << "P: " << P.num_rows() << " x " << P.num_cols() << std::endl;
		throw(std::runtime_error("Matrix sizes must be adequate."));
	}

	// todo: handle layouts and storage types correctly

	LILMatrix<double> RAP( P.num_cols(), P.num_cols() );

	// compute P^T A P = \sum P_ki * A_kl * P_lj
	for(size_t k = 0; k < P.num_rows(); ++k){
		for(auto it = P.begin(k); it !=  P.end(k); ++it){

			const size_t i = it.index();
			const double& P_ki = it.value();

			for(auto it = A.begin(k); it !=  A.end(k); ++it){

				const size_t l = it.index();
				const double& A_kl = it.value();

				const double P_ki_A_kl = P_ki * A_kl;

				for(auto it = P.begin(l); it !=  P.end(l); ++it){

					const size_t j = it.index();
					const double& P_lj = it.value();

					const double P_ki_A_kl_P_lj =  P_ki_A_kl * P_lj;
					if(P_ki_A_kl_P_lj != 0.0)
						RAP(i,j) += P_ki_A_kl_P_lj;
				}
			}
		}
	}

	PTAP = RAP;
}



#endif // ELL_MATRIX_CUDA_H
