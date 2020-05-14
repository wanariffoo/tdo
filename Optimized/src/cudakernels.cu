/*
    cudakernels.cu

    Developed for the master thesis project: GPU-accelerated Thermodynamic Topology Optimization
    Author: Wan Arif bin Wan Abhar
    Institution: Ruhr Universitaet Bochum
*/


#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include <ctime>
#include <iostream>
#include "../include/cudakernels.h"


#define CUDA_CALL( call )                                                                                          \
    {                                                                                                                  \
    cudaError_t err = call;                                                                                          \
    if ( cudaSuccess != err){                                                                                         \
        fprintf(stderr, "CUDA error for %s in %d of %s : %s.\n", #call , __LINE__ , __FILE__ ,cudaGetErrorString(err));exit(EXIT_FAILURE);}\
    }


using namespace std;


// Self-defined double-precision atomicAdd function for nvidia GPUs with Compute Capability 6 and below.
// Pre-defined atomicAdd() with double-precision does not work for pre-CC7 nvidia GPUs.
__device__ 
double atomicAdd_double(double* address, double val)
{
    unsigned long long int* address_as_ull =
                                (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val +
                                __longlong_as_double(assumed)));

    } while (assumed != old);

    return __longlong_as_double(old);
}


// determines 1-dimensional CUDA block and grid sizes based on the number of rows N
__host__ 
void calculateDimensions(size_t N, dim3 &gridDim, dim3 &blockDim)
{
    if ( N <= 1024 )
    {
        blockDim.x = 1024; blockDim.y = 1; blockDim.z = 1;
        gridDim.x  = 1; gridDim.y = 1; gridDim.z = 1;
    }
        
    else
    {
        blockDim.x = 1024; blockDim.y = 1; blockDim.z = 1;
        gridDim.x  = (int)ceil(N/blockDim.x)+1; gridDim.y = 1; gridDim.z = 1;
    }
}

// determines 2-dimensional CUDA block and grid sizes based on the number of rows N
__host__ void calculateDimensions2D(size_t Nx, size_t Ny, dim3 &gridDim, dim3 &blockDim)
{
    if ( Nx <= 32 && Ny <= 32)
    {
        blockDim.x = 32; blockDim.y = 32; blockDim.z = 1;
        gridDim.x  = 1; gridDim.y = 1; gridDim.z = 1;
    }
        
    else
    {
        blockDim.x = 32; blockDim.y = 32; blockDim.z = 1;
        gridDim.x  = (int)ceil(Nx/blockDim.x)+1; gridDim.y = (int)ceil(Ny/blockDim.y)+1; gridDim.z = 1;
    }


}

// calculates the DOF of a grid with dimensions
__host__ size_t calcDOF(size_t Nx, size_t Ny, size_t dim)
{
	return (Nx + 1) * (Ny + 1) * dim;
}


// returns value of an ELLPack matrix A at (x,y)
__device__
double valueAt(size_t x, size_t y, double* vValue, size_t* vIndex, size_t max_row_size)
{
    for(size_t k = 0; k < max_row_size; ++k)
    {
        if(vIndex[x * max_row_size + k] == y)
            return vValue[x * max_row_size + k];
    }

    return 0.0;
}

// returns value of a transposed ELLPack matrix A at (row,col)
__device__
double valueAt_(size_t row, size_t col, double* vValue, size_t* vIndex, size_t max_row_size, size_t num_rows)
{
    for(size_t k = 0; k < max_row_size; ++k)
    {
        if(vIndex[k * num_rows + row] == col)
            return vValue[k * num_rows + row];
    }

    return 0.0;
}


__device__
void setAt( size_t row, size_t col, double* vValue, size_t* vIndex, size_t max_row_size, size_t num_rows, double value )
{
	for(size_t k = 0; k < max_row_size; ++k)
    {	
        if(vIndex[k * num_rows + col] == row)
        {
            vValue[k * num_rows + col] = value;
                k = max_row_size; // to exit for loop
		}
    }
}


// a[] = 0.0
__global__
void setToZero(double* a, size_t num_rows)
{
	int id = blockDim.x * blockIdx.x + threadIdx.x;

	if ( id < num_rows )
		a[id] = 0.0;
}

// a = 1
__global__
void setToOne(double* a)
{
		*a = 1;
}

// norm = x.norm()
__global__ 
void norm_GPU(double* norm, double* x, size_t num_rows)
{
	int id = blockDim.x * blockIdx.x + threadIdx.x;

	if ( id == 0 )
		*norm = 0;
	__syncthreads();

	if ( id < num_rows )
	{
		#if __CUDA_ARCH__ < 600
		atomicAdd_double( norm, x[id]*x[id] );
		#else
		atomicAdd( norm, x[id]*x[id] );
		#endif

	}
	__syncthreads();

	if ( id == 0 )
		*norm = sqrt(*norm);
}

// a[] = 0, size_t
__global__
void setToZero(size_t* a, size_t num_rows)
{
	int id = blockDim.x * blockIdx.x + threadIdx.x;

	if ( id < num_rows )
		a[id] = 0.0;
}

// bool = true
__global__
void setToTrue( bool *foo )
{
	*foo = true;
}


// x = sqrt(x)
__global__
void sqrt_GPU(double *x)
{
	*x = sqrt(*x);
}

// sum = sum( x[n]*x[n] )
__global__ 
void sumOfSquare_GPU(double* sum, double* x, size_t n)
{
	int id = threadIdx.x + blockDim.x*blockIdx.x;
	int stride = blockDim.x*gridDim.x;
		
	__shared__ double cache[1024];
	
	double temp = 0.0;
	while(id < n)
	{
		temp += x[id]*x[id];
		
		id += stride;
	}
	
	cache[threadIdx.x] = temp;
	
	__syncthreads();
	
	// reduction
	unsigned int i = blockDim.x/2;
	while(i != 0){
		if(threadIdx.x < i){
			cache[threadIdx.x] += cache[threadIdx.x + i];
		}
		__syncthreads();
		i /= 2;
	}

	// reset id
	id = threadIdx.x + blockDim.x*blockIdx.x;

	// reduce sum from all blocks' cache
	if(threadIdx.x == 0)
	{
		#if __CUDA_ARCH__ < 600
			atomicAdd_double(sum, cache[0]);
		#else
			atomicAdd(sum, cache[0]);
		#endif
	}
}


__global__ 
void LastBlockSumOfSquare_GPU(double* sum, double* x, size_t n, size_t counter)
{
	int id = threadIdx.x + blockDim.x*blockIdx.x;
    
	if ( id >= counter*blockDim.x && id < n )
	{
		#if __CUDA_ARCH__ < 600
			atomicAdd_double(sum, x[id]*x[id]);
		#else
			atomicAdd(sum, x[id]*x[id]);
		#endif
	}
}

__host__
void norm_GPU(double* d_norm, double* d_x, size_t N, dim3 gridDim, dim3 blockDim)
{
	setToZero<<<1,1>>>( d_norm, 1);
    
    // getting the last block's size
    size_t lastBlockSize = N;
    size_t counter = 0;

    if ( N % gridDim.x == 0 ) {}
       

    else
    {
        while ( lastBlockSize >= gridDim.x)
        {
            counter++;
            lastBlockSize -= gridDim.x;
        }
    }

    // sum of squares for the full blocks
    sumOfSquare_GPU<<<gridDim.x - 1, blockDim>>>(d_norm, d_x, (gridDim.x - 1)*blockDim.x);

    // sum of squares for the last incomplete block
    LastBlockSumOfSquare_GPU<<<1, lastBlockSize>>>(d_norm, d_x, N, counter);
	
	sqrt_GPU<<<1,1>>>( d_norm );
	
}

//// DEBUG:
//// helper functions for debugging
__global__ 
void print_GPU(double* x)
{
	printf("[GPU] x = %e\n", *x);
}

__global__ 
void print_GPU(int* x)
{
	printf("[GPU] x = %d\n", *x);
}

__global__ 
void print_GPU(size_t* x)
{
	printf("[GPU] x = %lu\n", *x);
}

__global__ 
void print_GPU(bool* x)
{
	printf("[GPU] x = %d\n", *x);
}

__global__ void printLinearVector_GPU(size_t* x, size_t i, size_t num_rows, size_t num_cols)
{
        for ( int j = 0 ; j < num_cols ; j++ )
            printf("%lu ", x[j+i*num_cols]);

        printf("\n");
}

__global__ void printLinearVector_GPU(double* x, size_t i, size_t num_rows, size_t num_cols)
{
        for ( int j = 0 ; j < num_cols ; j++ )
            printf("%g ", x[j+i*num_cols]);

        printf("\n");
}

__host__ void printLinearVector(size_t* x, size_t num_rows, size_t num_cols)
{
	for(int i = 0 ; i < num_rows ; i++ )
	{
		printLinearVector_GPU<<<1,1>>>(x, i, num_rows, num_cols);
		cudaDeviceSynchronize();
	}

}

__host__ void printLinearVector(double* x, size_t num_rows, size_t num_cols)
{
	for(int i = 0 ; i < num_rows ; i++ )
	{
		printLinearVector_GPU<<<1,1>>>(x, i, num_rows, num_cols);
		cudaDeviceSynchronize();
	}

}

__global__ void print_GPU_(double* x, size_t i)
{
	printf("%d %g\n", i, x[i]);
}


__host__ void printVector(double* x, size_t num_rows)
{
	for ( int i = 0 ; i < num_rows ; i++ )
		print_GPU_<<<1,1>>>( x, i );

}

__global__ 
void printVector_GPU(double* x)
{
	int id = blockDim.x * blockIdx.x + threadIdx.x;

	printf("[GPU] x[%d] = %e\n", id, x[id]);
}

__global__
void printVector_GPU(double* x, size_t num_rows)
{
	int id = blockDim.x * blockIdx.x + threadIdx.x;

	if ( id < num_rows )
		printf("%d %e\n", id, x[id]);
}

__global__ 
void printVector_GPU(std::size_t* x, size_t num_rows)
{
	int id = blockDim.x * blockIdx.x + threadIdx.x;
	
	if ( id < num_rows )
		printf("%d %lu\n", id, x[id]);
	
}

__global__ 
void printVector_GPU(int* x)
{
	int id = blockDim.x * blockIdx.x + threadIdx.x;

	printf("[GPU] x[%d] = %d\n", id, x[id]);
}


__global__
void printELL_GPU(double* value, size_t* index, size_t max_row_size, size_t num_rows, size_t num_cols)
{
		for ( int i = 0 ; i < num_rows ; i++)
		{
			for ( int j = 0 ; j < num_cols ; j++)
			printf("%f ", valueAt(i, j, value, index, max_row_size) );

			printf("\n");
		}
}


__global__
void printELL_GPU_(double* value, size_t* index, size_t max_row_size, size_t num_rows, size_t num_cols)
{
		for ( int i = 0 ; i < num_rows ; i++)
		{
			for ( int j = 0 ; j < num_cols ; j++)
			printf("%g ", valueAt_(i, j, value, index, max_row_size, num_rows) );

			printf("\n");
		}
	
}


__global__
void printELLrow_GPU(size_t row, double* value, size_t* index, size_t max_row_size, size_t num_rows, size_t num_cols)
{
		for ( int j = 0 ; j < num_cols ; j++)
			printf("%.3f ", valueAt(row, j, value, index, max_row_size) );

		printf("\n");
	
}


__host__
void printELLrow(size_t lev, double* value, size_t* index, size_t max_row_size, size_t num_rows, size_t num_cols)
{

    for ( size_t i = 0 ; i < num_rows ; i++ )
    {
        printELLrow_GPU<<<1,1>>> (i, value, index, max_row_size, num_rows, num_cols);
        cudaDeviceSynchronize();    
    }
}


// prints matrix with size (num_rows, num_cols) that is stored in a transposed ELLPACK format
__global__
void printELLrow_GPU_(size_t row, double* value, size_t* index, size_t max_row_size, size_t num_rows, size_t num_cols)
{
		for ( int j = 0 ; j < num_cols ; j++)
			printf("%.3f ", valueAt_(row, j, value, index, max_row_size, num_rows) );

		printf("\n");
	
		
}

__host__
void printELLrow_(size_t lev, double* value, size_t* index, size_t max_row_size, size_t num_rows, size_t num_cols)
{
	for ( size_t i = 0 ; i < num_rows ; i++ )
    {
        printELLrow_GPU_<<<1,1>>> (i, value, index, max_row_size, num_rows, num_cols);
        cudaDeviceSynchronize();    
    }
}

// (scalar) a = b
__global__ 
void equals_GPU(double* a, double* b)
{
	*a = *b;
}


// x = a * b
__global__ 
void dotProduct_GPU(double* x, double* a, double* b, size_t num_rows)
{
	unsigned int id = threadIdx.x + blockDim.x*blockIdx.x;
	unsigned int stride = blockDim.x*gridDim.x;

	__shared__ double cache[1024];

	double temp = 0.0;

	// filling in the shared variable
	while(id < num_rows){
		temp += a[id]*b[id];

		id += stride;
	}

	cache[threadIdx.x] = temp;

	__syncthreads();

	// reduction
	unsigned int i = blockDim.x/2;
	while(i != 0){
		if(threadIdx.x < i){
			cache[threadIdx.x] += cache[threadIdx.x + i];
		}
		__syncthreads();
		i /= 2;
	}

	if(threadIdx.x == 0){
		#if __CUDA_ARCH__ < 600
			atomicAdd_double(x, cache[0]);
		#else
			atomicAdd(x, cache[0]);
		#endif		
	}
	__syncthreads();
}


__global__
void LastBlockDotProduct(double* dot, double* x, double* y, size_t starting_index)
{
	int id = threadIdx.x + blockDim.x*blockIdx.x + starting_index;
		
	#if __CUDA_ARCH__ < 600
		atomicAdd_double(dot, x[id]*y[id]);
	#else
		atomicAdd(dot, x[id]*y[id]);
	#endif		
	
}


// dot = a[] * b[]
__host__
void dotProduct(double* dot, double* a, double* b, size_t N, dim3 gridDim, dim3 blockDim)
{
    setToZero<<<1,1>>>( dot, 1 );

    // getting the last block's size
    size_t lastBlockSize = blockDim.x - ( (gridDim.x * blockDim.x ) - N );

	if ( N < blockDim.x)
	{
		LastBlockDotProduct<<<1, N>>>( dot, a, b, 0 );
	}

	else
	{
		// dot products for the full blocks
		dotProduct_GPU<<<gridDim.x - 1, blockDim>>>(dot, a, b, (gridDim.x - 1)*blockDim.x );
		
		// dot products for the last incomplete block
		LastBlockDotProduct<<<1, lastBlockSize>>>(dot, a, b, ( (gridDim.x - 1) * blockDim.x ) );
	}

}

// x = y / z
__global__
void divide_GPU(double *x, double *y, double *z)
{
	*x = *y / *z;
}
 
// x += y
__global__ void add_GPU(double *x, double *y)
{
	*x += *y;
}

// x -= y
__global__ void minus_GPU(double *x, double *y)
{
	*x -= *y;
}


// x += c
__global__
void addVector_GPU(double *x, double *c, size_t num_rows)
{
	int id = blockDim.x * blockIdx.x + threadIdx.x;

	if ( id < num_rows )
		x[id] += c[id];
}

// a = b
__global__ 
void vectorEquals_GPU(double* a, double* b, size_t num_rows)
{
	int id = blockDim.x * blockIdx.x + threadIdx.x;

	if ( id < num_rows )
		a[id] = b[id];
}





////////////////////////////////////////////
// ASSEMBLER
////////////////////////////////////////////
  

__host__ 
vector<vector<size_t>> applyBC(vector<size_t> N, size_t numLevels, size_t bc_case, size_t dim)
{
	vector<vector<size_t>> bc_index(numLevels);
	
	vector<size_t> nodesPerDim;

	for( int i = 0 ; i < N.size() ; i++ )
		nodesPerDim.push_back(N[i]+1);
	
	if ( bc_case == 0 )
	{
		// base level
		size_t totalNodes2D = nodesPerDim[0]*nodesPerDim[1];

		for ( int i = 0 ; i < nodesPerDim[1] ; i++ )
		{
			bc_index[0].push_back(i*nodesPerDim[0]*dim);

			if ( dim == 3 )
			{
				for ( int j = 1 ; j < nodesPerDim[2] ; j++ )
					bc_index[0].push_back(i*nodesPerDim[0]*dim + totalNodes2D*3*j);
			}
		}

		// y-direction boundary condition at bottom right node
		bc_index[0].push_back(dim*N[0] + 1 );

		if ( dim == 3 )
		{
			for ( int j = 1 ; j < nodesPerDim[2] ; j++ )
				bc_index[0].push_back(dim*N[0] + 1 + totalNodes2D*3*j);
		}
		

		// finer levels
		for ( int lev = 1 ; lev < numLevels ; lev++ )
		{
			for( int i = 0 ; i < N.size() ; i++ )
				nodesPerDim[i] = 2*nodesPerDim[i] - 1;

			totalNodes2D = nodesPerDim[0]*nodesPerDim[1];

			for ( int i = 0 ; i < nodesPerDim[1] ; i++ )
			{
				bc_index[lev].push_back(i*nodesPerDim[0]*dim);

				if ( dim == 3 )
				{
					for ( int j = 1 ; j < nodesPerDim[2] ; j++ )
						bc_index[lev].push_back(i*nodesPerDim[0]*dim + totalNodes2D*3*j);

				}

			}

			// y-direction boundary condition at bottom right node
			bc_index[lev].push_back(nodesPerDim[0]*dim - (dim-1));
			
			if ( dim == 3 )
			{
				for ( int j = 1 ; j < nodesPerDim[2] ; j++ )
					bc_index[lev].push_back(dim*nodesPerDim[0] - (dim-1) + totalNodes2D*3*j);

			}

		}
	}

	else if ( bc_case == 1 )
	{
		if ( N.size() < 3 )
			throw(runtime_error("Error : Boundary condition case 1 is not set up yet for 2D"));

		// base level
		size_t totalNodes2D = nodesPerDim[0]*nodesPerDim[1];

		// plane where u2 = 0
		for ( int i = 0 ; i < totalNodes2D ; i++ )
			bc_index[0].push_back(i*dim + 2);

		// 2 points with pinned BC
		bc_index[0].push_back( totalNodes2D*3*N[2] );
		bc_index[0].push_back( totalNodes2D*3*N[2] + 1 );
		bc_index[0].push_back( totalNodes2D*3*N[2] + 2 );
		bc_index[0].push_back( totalNodes2D*3*N[2] + (N[0]+1) * (N[1]) * 3 );
		bc_index[0].push_back( totalNodes2D*3*N[2] + 1 + (N[0]+1) * (N[1]) * 3 );
		bc_index[0].push_back( totalNodes2D*3*N[2] + 2 + (N[0]+1) * (N[1]) * 3 );

		// finer levels
		for ( int lev = 1 ; lev < numLevels ; lev++ )
		{
			for( int i = 0 ; i < N.size() ; i++ )
			{
				nodesPerDim[i] = 2*nodesPerDim[i] - 1;
				N[i] *= 2;
			}

			totalNodes2D = nodesPerDim[0]*nodesPerDim[1];

			// plane where u2 = 0
			for ( int i = 0 ; i < totalNodes2D ; i++ )
				bc_index[lev].push_back(i*dim + 2);

			// 2 points with pinned BC
			bc_index[lev].push_back( totalNodes2D*3*N[2] );
			bc_index[lev].push_back( totalNodes2D*3*N[2] + 1 );
			bc_index[lev].push_back( totalNodes2D*3*N[2] + 2 );
			bc_index[lev].push_back( totalNodes2D*3*N[2] + (N[0]+1) * (N[1]) * 3 );
			bc_index[lev].push_back( totalNodes2D*3*N[2] + 1 + (N[0]+1) * (N[1]) * 3 );
			bc_index[lev].push_back( totalNodes2D*3*N[2] + 2 + (N[0]+1) * (N[1]) * 3 );
			
		}
	}

	return bc_index;
}


__host__ 
void applyLoad(vector<double> &b, vector<size_t> N, size_t numLevels, size_t bc_case, size_t dim, double force)
{
	
	if ( bc_case == 0 )
	{
		vector<size_t> nodesPerDim;

		for ( int i = 0 ; i < N.size() ; i++)
			nodesPerDim.push_back(N[i]+1);

		size_t index = 0;
		
		for ( int lev = 0 ; lev < numLevels - 1 ; lev++)
		{
			for ( int i = 0 ; i < N.size() ; i++)
			nodesPerDim[i] = 2*nodesPerDim[i] - 1;

		}

		index = dim * nodesPerDim[0] * ( nodesPerDim[1] - 1 ) + 1;
		
		b[index] = force;
		
		if ( dim == 3 )
		{
			
			for ( int i = 1 ; i < nodesPerDim[2] ; i++ )
			{
				index = index + (nodesPerDim[0]*nodesPerDim[1])*dim;
				b[index] = force;
				
			}
		}
	}

	else if ( bc_case == 1 )
	{
		if ( N.size() < 3 )
			throw(runtime_error("Error : Load case 1 is not set up yet for 2D"));
		
		// obtaining the finest grid's number of elements on the x-axis
		size_t Nx_fine = N[0];

		for ( int lev = 0 ; lev < numLevels - 1 ; lev++)
			Nx_fine *= 2;

		size_t index = (Nx_fine+1)*dim - 2;
		b[index] = force;
	}
}




// adds local stiffness matrix of an element to the global stiffness matrix
__global__
void assembleGlobalStiffness_GPU(
    size_t numElements,     // total number of elements
    size_t dim,             // dimension
	double* chi,			// the updated design variable value of each element
	double* A_local,      	// local stiffness matrix
	size_t num_rows_l,      // local stiffness matrix's number of rows
    double* value,        	// global element's ELLPACK value vector
    size_t* index,        	// global element's ELLPACK index vector
    size_t max_row_size,  	// global element's ELLPACK maximum row size
    size_t num_rows,      	// global element's ELLPACK number of rows
    size_t* node_index,     // vector that contains the corresponding global indices of the node's local indices
	size_t p					
)        
{
	int id = threadIdx.x + blockIdx.x*blockDim.x;

	if ( id < numElements )
	{
		int numNodesPerElement = pow(2,dim);
		for ( int row = 0 ; row < num_rows_l ; row++ )
		{
			int y = dim*node_index[ (row/dim) + (id*numNodesPerElement) ] + ( row % dim );

			for ( int col = 0 ; col < num_rows_l ; col++ )
			{
				int x = dim*node_index[ (col/dim) + (id*numNodesPerElement) ] + ( col % dim );

				atomicAddAt( x, y, value, index, max_row_size, num_rows, pow(chi[id],p)*A_local[ ( col + row*num_rows_l ) ] );
			}
		}
	}
}



// applies boundary condition on the global stiffness matrix (2d case) where the affected row/column is set to '0'' and its diagonal to '1'
__global__
void applyMatrixBC2D_GPU(double* value, size_t* index, size_t max_row_size, size_t* bc_index, size_t num_rows, size_t bc_size, size_t Nx, size_t Ny, size_t dim)
{
	int id = threadIdx.x + blockIdx.x*blockDim.x;

	if ( id < bc_size )
	{
		// assigning each thread to a single bc index
		size_t bc = bc_index[id];

		// setting the row entries to '0'
		for ( int i = 0 ; i < max_row_size ; i++ )
		{
			value[ bc + i*num_rows ] = 0.0;
		}

		// setting the diagonal to '1'

		// setting the column entries to '0' through the neighbouring nodes

		int base_id = (bc - bc%dim);
		
		bool south = ( bc  >= (Nx + 1)*dim );
		bool north = ( bc < (Nx+1)*(Ny)*dim );
		bool west = ( (bc) % ((Nx + 1)*dim) >= dim );
		bool east = ( (base_id) % ((Nx*dim) + (base_id/(2*(Nx+1)))*dim*(Nx+1)) != 0 );

		// south-west
		if ( south && west )
		{
			for(int i = 0 ; i < dim ; i++)
				setAt( bc, bc - (dim * (Nx+1)) - dim + i, value, index, max_row_size, num_rows, 0.0 );
		}

		// south
		if ( south )
		{
			for(int i = 0 ; i < dim ; i++)
				setAt( bc, bc - (dim * (Nx+1)) + i, value, index, max_row_size, num_rows, 0.0 );
		}

		// south-east
		if ( south && east )
		{
			for(int i = 0 ; i < dim ; i++)
				setAt( bc, bc - (dim * (Nx+1)) + dim + i, value, index, max_row_size, num_rows, 0.0 );
		}

		// west
		if ( west )
		{
			for(int i = 0 ; i < dim ; i++)
				setAt( bc, bc - dim + i, value, index, max_row_size, num_rows, 0.0 );
		}

		// origin
		{
			// setting the diagonal to '1' 
			setAt( bc, bc, value, index, max_row_size, num_rows, 1.0 );

			// and other DOFs on the node to '0'
			for ( int i = 1 ; i < dim ; i++)
				setAt( bc, bc + i, value, index, max_row_size, num_rows, 0.0 );
		}

		// east
		if ( base_id == 0 || east )
		{
			for ( int i = 0 ; i < dim ; i++)
				setAt( bc, bc + dim + i, value, index, max_row_size, num_rows, 0.0 );
		}

		// north-west
		if ( north && west )
		{
			for(int i = 0 ; i < dim ; i++)
				setAt( bc, bc + (dim * (Nx+1)) - dim + i, value, index, max_row_size, num_rows, 0.0 );
		}

		// north
		if ( north )
		{
			for(int i = 0 ; i < dim ; i++)
				setAt( bc, bc + (dim * (Nx+1)) + i, value, index, max_row_size, num_rows, 0.0 );
		}
		
		// north-east
		if ( base_id == 0 || id < (Nx+1)*(Ny)*dim && east )
		{
			for(int i = 0 ; i < dim ; i++)
				setAt( bc, bc + (dim * (Nx+1)) + dim + i, value, index, max_row_size, num_rows, 0.0 );
		}
	}
}

// applies boundary condition on the global stiffness matrix (3d case) where the affected row/column is set to '0'' and its diagonal to '1'
__global__
void applyMatrixBC3D_GPU(double* value, size_t* index, size_t max_row_size, size_t* bc_index, size_t num_rows, size_t bc_size, size_t Nx, size_t Ny, size_t Nz, size_t dim)
{
	int id = threadIdx.x + blockIdx.x*blockDim.x;

	if ( id < bc_size )
	{
		// assigning each thread to a single bc index
		size_t bc = bc_index[id];


		// setting the row entries to '0'
		for ( int i = 0 ; i < max_row_size ; i++ )
		{
			value[ bc + i*num_rows ] = 0.0;
		}

		// setting the column entries to '0' through the neighbouring nodes
		size_t base_id = (id - id%dim);
		size_t gridsize_2D = (Nx+1)*(Ny+1)*dim;

		bool prev_layer = (bc >= (Nx+1)*(Ny+1)*dim);
		bool south = ((bc) % ((Nx + 1)*(Ny + 1)*dim)  >= (Nx + 1)*dim);
		bool north = ((bc) % ((Nx + 1)*(Ny + 1)*dim) < (Nx+1)*(Ny)*dim);
		bool west = ((bc) % ((Nx + 1)*dim) >= dim);
		bool east = ((base_id) % ((Nx*dim) + (base_id/(3*(Nx+1)))*dim*(Nx+1)) != 0);

	//// previous layer

		// south-west
		if ( prev_layer && south && west )
		{
			for(int i = 0 ; i < dim ; i++)
				setAt( bc, bc - (dim * (Nx+1)) - dim + i - gridsize_2D, value, index, max_row_size, num_rows, 0.0 );
		}

		// south
		if ( prev_layer && south )
		{
			for(int i = 0 ; i < dim ; i++)
				setAt( bc, bc - (dim * (Nx+1)) + i - gridsize_2D, value, index, max_row_size, num_rows, 0.0 );
		}

		// south-east
		if ( prev_layer && south && east )
		{	
			for(int i = 0 ; i < dim ; i++)
				setAt( bc, bc - (dim * (Nx+1)) + dim + i - gridsize_2D, value, index, max_row_size, num_rows, 0.0 );
		}

		// west
		if ( prev_layer && west )
		{
			for(int i = 0 ; i < dim ; i++)
				setAt( bc, bc - dim + i - gridsize_2D, value, index, max_row_size, num_rows, 0.0 );
		}

		// origin
		if ( prev_layer )
		{
			for(int i = 0 ; i < dim ; i++)
				setAt( bc, bc + i - gridsize_2D, value, index, max_row_size, num_rows, 0.0 );
		}

		// east
		if ( prev_layer && east )
		{
			for(int i = 0 ; i < dim ; i++)
				setAt( bc, bc + dim + i - gridsize_2D, value, index, max_row_size, num_rows, 0.0 );
		}

		// north-west
		if ( prev_layer && north && west )
		{
			for(int i = 0 ; i < dim ; i++)
				setAt( bc, bc + (dim * (Nx+1)) - dim + i - gridsize_2D, value, index, max_row_size, num_rows, 0.0 );
		}

		// north
		if ( prev_layer && north )
		{
			for(int i = 0 ; i < dim ; i++)
				setAt( bc, bc + (dim * (Nx+1)) + i - gridsize_2D, value, index, max_row_size, num_rows, 0.0 );
		}

		// north-east
		if ( prev_layer && north && east )
		{
			for(int i = 0 ; i < dim ; i++)
				setAt( bc, bc + (dim * (Nx+1)) + dim + i - gridsize_2D, value, index, max_row_size, num_rows, 0.0 );
		}

	//// current layer

		// south-west
		if ( south && west )
		{
			for(int i = 0 ; i < dim ; i++)
				setAt( bc, bc - (dim * (Nx+1)) - dim + i, value, index, max_row_size, num_rows, 0.0 );
		}

		// south
		if ( south )
		{
			for(int i = 0 ; i < dim ; i++)
				setAt( bc, bc - (dim * (Nx+1)) + i, value, index, max_row_size, num_rows, 0.0 );
		}

		// south-east
		if ( south && east )
		{
			for(int i = 0 ; i < dim ; i++)
				setAt( bc, bc - (dim * (Nx+1)) + dim + i, value, index, max_row_size, num_rows, 0.0 );
		}

		// west
		if ( west )
		{
			for(int i = 0 ; i < dim ; i++)
				setAt( bc, bc - dim + i, value, index, max_row_size, num_rows, 0.0 );
		}

		// origin
		{
			// setting the diagonal to '1' 
			setAt( bc, bc, value, index, max_row_size, num_rows, 1.0 );

			// and other DOFs on the node to '0'
			for ( int i = 1 ; i < dim ; i++)
				setAt( bc, bc + i, value, index, max_row_size, num_rows, 0.0 );
		}

		// east
		if ( base_id == 0 || east )
		{
			for ( int i = 0 ; i < dim ; i++)
				setAt( bc, bc + dim + i, value, index, max_row_size, num_rows, 0.0 );
		}

		// north-west
		if ( north && west )
		{
			for(int i = 0 ; i < dim ; i++)
				setAt( bc, bc + (dim * (Nx+1)) - dim + i, value, index, max_row_size, num_rows, 0.0 );
		}

		// north
		if ( north )
		{
			for(int i = 0 ; i < dim ; i++)
				setAt( bc, bc + (dim * (Nx+1)) + i, value, index, max_row_size, num_rows, 0.0 );
		}
		
		// north-east
		if ( base_id == 0 || (north && east ) )
		{
			for(int i = 0 ; i < dim ; i++)
				setAt( bc, bc + (dim * (Nx+1)) + dim + i, value, index, max_row_size, num_rows, 0.0 );
		}

	//// next layer

		// south-west
		if ( prev_layer && south && west )
		{
			for(int i = 0 ; i < dim ; i++)
				setAt( bc, bc - (dim * (Nx+1)) - dim + i + gridsize_2D, value, index, max_row_size, num_rows, 0.0 );
		}

		// south
		if ( prev_layer && south )
		{
			for(int i = 0 ; i < dim ; i++)
				setAt( bc, bc - (dim * (Nx+1)) + i + gridsize_2D, value, index, max_row_size, num_rows, 0.0 );
		}

		// south-east
		if ( prev_layer && south && east )
		{	
			for(int i = 0 ; i < dim ; i++)
				setAt( bc, bc - (dim * (Nx+1)) + dim + i + gridsize_2D, value, index, max_row_size, num_rows, 0.0 );
		}

		// west
		if ( prev_layer && west )
		{
			for(int i = 0 ; i < dim ; i++)
				setAt( bc, bc - dim + i + gridsize_2D, value, index, max_row_size, num_rows, 0.0 );
		}

		// origin
		if ( prev_layer )
		{
			for(int i = 0 ; i < dim ; i++)
				setAt( bc, bc + i + gridsize_2D, value, index, max_row_size, num_rows, 0.0 );
		}

		// east
		if ( prev_layer && east )
		{
			for(int i = 0 ; i < dim ; i++)
				setAt( bc, bc + dim + i + gridsize_2D, value, index, max_row_size, num_rows, 0.0 );
		}

		// north-west
		if ( prev_layer && north && west )
		{
			for(int i = 0 ; i < dim ; i++)
				setAt( bc, bc + (dim * (Nx+1)) - dim + i + gridsize_2D, value, index, max_row_size, num_rows, 0.0 );
		}

		// north
		if ( prev_layer && north )
		{
			for(int i = 0 ; i < dim ; i++)
				setAt( bc, bc + (dim * (Nx+1)) + i + gridsize_2D, value, index, max_row_size, num_rows, 0.0 );
		}

		// north-east
		if ( prev_layer && north && east )
		{
			for(int i = 0 ; i < dim ; i++)
				setAt( bc, bc + (dim * (Nx+1)) + dim + i + gridsize_2D, value, index, max_row_size, num_rows, 0.0 );
		}
	}
}

// applies boundary condition on the prolongation matrix where the affected row/column is set to '0'' and its diagonal to '1'
__global__
void applyProlMatrixBC_GPU(	double* value, size_t* index, size_t max_row_size, 
							size_t* bc_index, size_t* bc_index_, 
							size_t num_rows, size_t num_rows_, 
							size_t bc_size, size_t bc_size_)
{
	int id = blockDim.x * blockIdx.x + threadIdx.x;
	
	if ( id < num_rows )
	{
		for ( int row = 0 ; row < max_row_size ; row++ )
		{
			for ( int i = 0 ; i < bc_size_ ; i++ )
			{
				size_t bc_row = bc_index_[i];
				
				if ( value[ id + row*num_rows ] != 1 && index[id + row*num_rows] == bc_row )
						value[id + row*num_rows ] = 0;
			}
		}
	}
}

// input the coarse node's "index" to obtain the node's corresponding fine node index
__device__
size_t getFineNode_GPU(size_t index, size_t Nx, size_t Ny, size_t Nz, size_t dim)
{
	
	if ( dim == 3 )
	{	
		size_t twoDimSize = (Nx+1)*(Ny+1);
		size_t baseindex = index % twoDimSize;
		size_t base_idx = baseindex % (Nx+1);
		size_t fine2Dsize = (2*Nx+1)*(2*Ny+1);
		size_t multiplier = index/twoDimSize;
		
		return 2*base_idx + (baseindex/(Nx+1))*2*(2*Nx + 1) + 2*fine2Dsize*multiplier;
	}

	else
		return (2 * (index / (Nx + 1)) * (2*Nx + 1) + 2*( index % (Nx+1)) );
}


// ////////////////////////////////////////////
// // SMOOTHERS
// ////////////////////////////////////////////

__global__ void Jacobi_Precond_GPU(double* c, double* value, size_t* index, size_t max_row_size, double* r, size_t num_rows, double damp){

	int id = blockDim.x * blockIdx.x + threadIdx.x;

	// B = damp / diag(A);
	if ( id < num_rows )
		c[id] = r[id] * damp / valueAt_(id, id, value, index, max_row_size, num_rows);

}


// ////////////////////////////////////////////
// // SOLVER
// ////////////////////////////////////////////

__global__ 
void checkIterationConditions(bool* foo, size_t* step, double* res, double* res0, double* m_minRes, double* m_minRed, size_t m_maxIter)
{
	if ( *res > *m_minRes && *res > *m_minRed*(*res0) && (*step) <= m_maxIter )
	{
		*foo = true;
	}

	else
		*foo = false;
}


__global__
void checkIterationConditionsBS(bool* foo, size_t* step, size_t m_maxIter, double* res, double* m_minRes)
{
	if ( *res > 1e-12 && (*step) <= m_maxIter )
	{
		*foo = true;
	}

	else
		*foo = false;
}

__global__ 
void printInitialResult_GPU(double* res0, double* m_minRes, double* m_minRed)
{
	printf("    0    %e    %9.3e      -----        --------      %9.3e    \n", *res0, *m_minRes, *m_minRed);
}

/// r = b - A*x
__global__ 
void ComputeResiduum_GPU(	
	const std::size_t num_rows, 
	const std::size_t max_row_size,
	const double* value,
	const std::size_t* index,
	const double* x,
	double* r,
	double* b)
{
	int id = blockDim.x * blockIdx.x + threadIdx.x;

	if ( id < num_rows )
    {
        double sum = 0;
        for ( int n = 0 ; n < max_row_size; n++ )
        {
			unsigned int offset = id + n*num_rows;
			// sum += value[offset] * x[ index[offset] ];
            sum += value[offset] * __ldg( &x[ index[offset] ] );
        }
        r[id] = b[id] - sum;
    }
}


/// r = r - A*x
__global__ 
void UpdateResiduum_GPU(
	const std::size_t num_rows, 
	const std::size_t max_row_size,
	const double* value,
	const std::size_t* index,
	const double* x,
	double* r)
{
  	int id = blockDim.x * blockIdx.x + threadIdx.x;

	  if ( id < num_rows )
	  {
		  double sum = 0;
		  for ( int n = 0 ; n < max_row_size; n++ )
		  {
			  unsigned int offset = id + n*num_rows;
			  sum += value[offset] * __ldg( &x[ index[offset] ] );
		  }
		  r[id] = r[id] - sum;
	  }
}


// Ax = r for transposed ELLPACK format
__global__ void Apply_GPU (
	const std::size_t num_rows, 
	const std::size_t max_row_size,
	const double* value,
	const std::size_t* index,
	const double* x,
	double* r)
{
    int id = blockDim.x * blockIdx.x + threadIdx.x;

    if ( id < num_rows )
    {
        double sum = 0;
        for ( int n = 0 ; n < max_row_size; n++ )
        {
			unsigned int offset = id + n*num_rows;
			sum += value[offset] * x[ index[offset] ];
        }
        r[id] = sum;
    }
}


/// r = A^T * x for transposed ELLPACK format
/// NOTE: This kernel should be run with A's number of rows as the number of threads
__global__ 
void ApplyTransposed_GPU(	
	const std::size_t num_rows, 
	const std::size_t max_row_size,
	const double* value,				// A's ELL value array
	const std::size_t* index,			// A's ELL index array
	const double* x,					
	double* r)
{
	int id = blockDim.x * blockIdx.x + threadIdx.x;

	if ( id < num_rows )
	{
		for ( int n = 0; n < max_row_size; n++ )
		{
			int col = index [ id + n*num_rows ];
			double val = value [ id + n*num_rows ];

			#if __CUDA_ARCH__ < 600
				atomicAdd_double( &r[col], val*x[id] );
			#else
				atomicAdd( &r[col], val*x[id] );
			#endif		

		}
	}
}


// outputs result in the terminal
__global__ 
void printResult_GPU(size_t* step, double* res, double* m_minRes, double* lastRes, double* res0, double* m_minRed)
{
	if(*step < 10)
	printf("    %d    %e    %9.3e    %9.3e    %e    %9.3e    \n", *step, *res, *m_minRes, (*res)/(*lastRes), (*res)/(*res0), *m_minRed);

	else
	printf("   %d    %e    %9.3e    %9.3e    %e    %9.3e    \n", *step, *res, *m_minRes, (*res)/(*lastRes), (*res)/(*res0), *m_minRed);
}

// increases the iteration step
__global__ void addStep(size_t* step){

	++(*step);
}

// p = z + p * (rho / rho_old);
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
		if(*d_step == 1)
		{ 
			d_p[id] = d_z[id]; 
		}
		
		else
		{
			// p *= (rho / rho_old)
			d_p[id] = d_p[id] * ( *d_rho / (*d_rho_old) );
		
			// p += z;
			d_p[id] = d_p[id] + d_z[id];
		}
	}
}


// d_alpha = *d_rho / ( d_p * d_z )
__host__
void calculateAlpha(
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

	// alpha_temp = ( p * z )
	dotProduct(d_alpha_temp, d_p, d_z, num_rows, gridDim, blockDim);

	// d_alpha = *d_rho / (*alpha_temp)
	divide_GPU<<<1,1>>>(d_alpha, d_rho, d_alpha_temp);

}


// x = x + alpha * p
__global__ 
void axpy_GPU(double* d_x, double* d_alpha, double* d_p, size_t num_rows)
{
	int id = blockDim.x * blockIdx.x + threadIdx.x;

	if ( id < num_rows )
		d_x[id] += (*d_alpha * d_p[id]);
}

// x = x - alpha * p
__global__ 
void axpy_neg_GPU(double* d_x, double* d_alpha, double* d_p, size_t num_rows)
{
	int id = blockDim.x * blockIdx.x + threadIdx.x;

	if ( id < num_rows )
		d_x[id] = d_x[id] - (*d_alpha * d_p[id]);
}


//// TDO

// calculates the driving force of all elements
// one thread computes one element
// df[] = 0.5 * p * pow(chi[], p-1) / local_volume * u[]^T * A * u[]
__global__
void calcDrivingForce(	double *df, 			// driving force
						double *u, 				// displacement vector
						double* chi, 			// design variable
						double p, 				// penalization parameter
						size_t* node_index, 	// node index array
						double* d_A_local, 		// local stiffness matrix
						size_t num_rows, 		// num_rows of local stiffness matrix
						size_t dim, 			// dimension
						double local_volume, 	
						size_t numElements)
{
	int id = blockDim.x * blockIdx.x + threadIdx.x;

	if ( id < numElements)
	{
		double temp[24];
		size_t numNodesPerElement = pow(2,dim);
		
		df[id] = 0;
		for ( int n = 0; n < num_rows; n++ )
		{
			temp[n]=0;
			for ( int m = 0; m < num_rows; m++)
			{
				// converts local node to global node
				int global_col = ( node_index [ (m / dim) + id*numNodesPerElement ] * dim ) + ( m % dim ); 
				temp[n] += u[global_col] * d_A_local[ n + m*num_rows ];
			}
		}
		
		for ( int n = 0; n < num_rows; n++ )
		{
			int global_col = ( node_index [ (n / dim) + id*numNodesPerElement ] * dim ) + ( n % dim );
			df[id] += temp[n] * u[global_col];
		}
	
		df[id] *= 0.5 * p * pow(chi[id], p-1) / local_volume;

	}

}

// sum = sum(x)
// n = size of x vector
__global__ 
void sumOfVector_GPU(double* sum, double* x, size_t n)
{
    int id = blockDim.x * blockIdx.x + threadIdx.x;
	int stride = blockDim.x*gridDim.x;
    
	__shared__ double cache[1024];
    cache[threadIdx.x] = 0;
    
	double temp = 0.0;
	while(id < n)
	{
		temp += x[id];
		id += stride;
	}
	
    cache[threadIdx.x] = temp;
    
	__syncthreads();
	
	// reduction
	unsigned int i = blockDim.x/2;
	while(i != 0){
		if(threadIdx.x < i){
			cache[threadIdx.x] += cache[threadIdx.x + i];
		}
		__syncthreads();
		i /= 2;
	}

	// reduce sum from all blocks' cache
	if(threadIdx.x == 0)
	{
		#if __CUDA_ARCH__ < 600
			atomicAdd_double(sum, cache[0]);
		#else
			atomicAdd(sum, cache[0]);
		#endif		
	}
		
}


// laplacian for both 2d and 3d cases
// for 2d, Nz has to be predefined to '1'
__device__
double laplacian_GPU( double *array, size_t ind, size_t Nx, size_t Ny, size_t Nz, double h )
{

	bool east = ( (ind + 1) % Nx != 0 );
	bool north = ( ind + Nx < Nx*Ny );
	bool west = ( ind % Nx != 0 );
	bool south = ( ind >= Nx );
	bool previous_layer = (ind >= Nx*Ny);
	bool next_layer = (ind < Nx*Ny*(Nz-1));
	
	double value = -4.0 * array[ind];
	
    // east element
    if ( east )
        value += 1.0 * array[ind + 1];
	else
		value += 1.0 * array[ind];
	
    
    // north element
    if ( north )
        value += 1.0 * array[ind + Nx];
	else
		value += 1.0 * array[ind];

    // west element
    if ( west )
        value += 1.0 * array[ind - 1];
	else
		value += 1.0 * array[ind];

    // south element
    if ( south )
        value += 1.0 * array[ind - Nx];
	else
		value += 1.0 * array[ind];

	// if 3D
	if (Nz > 0)
	{
		value -= 2.0 * array[ind];

		// previous layer's element
		if ( previous_layer )
			value += 1.0 * array[ind - (Nx*Ny)];
		else
			value += 1.0 * array[ind];
		

		if ( next_layer )
			value += 1.0 * array[ind + (Nx*Ny)];
		else
			value += 1.0 * array[ind];
		
	}

    return value/(h*h);
}

__global__ 
void calcLambdaUpper(double *df_array, double *max, int *mutex, double* beta, double *chi, double* eta, int Nx, int Ny, int Nz, unsigned int numElements, double h)
{
	unsigned int index = threadIdx.x + blockIdx.x*blockDim.x;
	unsigned int stride = gridDim.x*blockDim.x;
	unsigned int offset = 0;

	__shared__ double cache[1024];

    *max = -1.0e9;
	*mutex = 0;
    double temp = -1.0e9;
    
	while(index + offset < numElements){
        
        temp = fmaxf(temp, ( df_array[index + offset] + ( *beta * laplacian_GPU( chi, index, Nx, Ny, Nz, h ) ) ) );
        // temp = fmaxf(temp, ( df_array[index + offset] + *eta ) );
        
		offset += stride;
	}
    
	cache[threadIdx.x] = temp;
	__syncthreads();

	// reduction
	unsigned int i = blockDim.x/2;
	while(i != 0){
		if(threadIdx.x < i){
			cache[threadIdx.x] = fmaxf(cache[threadIdx.x], cache[threadIdx.x + i]);
		}

		__syncthreads();
		i /= 2;
	}

	if(threadIdx.x == 0){
		while(atomicCAS(mutex,0,1) != 0);  //lock
		*max = fmaxf(*max, cache[0]);
		atomicExch(mutex, 0);  //unlock
    }
}


__global__ 
void calcLambdaLower(double *df_array, double *min, int *mutex, double* beta, double *chi, double* eta, int Nx, int Ny, int Nz, unsigned int numElements, double h)
{
	unsigned int index = threadIdx.x + blockIdx.x*blockDim.x;
	unsigned int stride = gridDim.x*blockDim.x;
	unsigned int offset = 0;

	__shared__ double cache[1024];

    *min = 1.0e9;
	*mutex = 0;
    double temp = 1.0e9;
    
	if ( index < numElements )
	{

		while(index + offset < numElements){
			
			temp = fminf(temp, ( df_array[index + offset] + ( *beta * laplacian_GPU( chi, index, Nx, Ny, Nz, h ) ) ) );
			// temp = fminf(temp, ( df_array[index + offset] - *eta ) );
			offset += stride;
		}
		
		cache[threadIdx.x] = temp;
		__syncthreads();


		// reduction
		unsigned int i = blockDim.x/2;
		while(i != 0){
			if(threadIdx.x < i){
				cache[threadIdx.x] = fminf(cache[threadIdx.x], cache[threadIdx.x + i]);
			}

			__syncthreads();
			i /= 2;
		}

		if(threadIdx.x == 0){
			while(atomicCAS(mutex,0,1) != 0);  //lock
			*min = fminf(*min, cache[0]);
			atomicExch(mutex, 0);  //unlock
		}
	}
    
}

__global__
void calcChiTrial(   
    double *chi, 
    double *df, 
    double *lambda_trial, 
    double del_t,
    double* eta,
    double* beta,
    double* chi_trial,
    size_t Nx,
    size_t Ny,
    size_t Nz,
    size_t numElements,
	double h
)
{
    unsigned int id = threadIdx.x + blockIdx.x*blockDim.x;

    if ( id < numElements )
    {
		double del_chi;
		
        del_chi = ( del_t / *eta ) * ( df[id] - *lambda_trial + (*beta)*( laplacian_GPU( chi, id, Nx, Ny, Nz, h ) ) );

        if ( del_chi + chi[id] > 1 )
        chi_trial[id] = 1;
        
        else if ( del_chi + chi[id] < 1e-9 )
        chi_trial[id] = 1e-9;
        
        else
        chi_trial[id] = del_chi + chi[id];
        
    }
}


__global__
void calcLambdaTrial(double *rho_trial, double rho, double *lambda_l, double *lambda_u, double *lambda_trial)
{
    if ( *rho_trial > rho )
        *lambda_l = *lambda_trial;

    else
        *lambda_u = *lambda_trial;

    *lambda_trial = 0.5 * ( *lambda_l + *lambda_u );
}


__global__ void calcRhoTrial(double* rho_tr, double local_volume, size_t numElements)
{
	double total_volume = local_volume * numElements;

	*rho_tr *= local_volume;
	*rho_tr /= total_volume;

}



// calculate the average weighted driving force, p_w
__global__ 
void calcP_w_GPU(double* p_w, double* df, double* uTAu, double* chi, int p, double local_volume, size_t numElements)
{
	unsigned int id = threadIdx.x + blockIdx.x*blockDim.x;

	__shared__ double int_g_p[1024];
	__shared__ double int_g[1024];

	if( id < numElements)
	{
		df[id] = uTAu[id] * ( local_volume / (2*local_volume) ) * p * pow(chi[id], p - 1);

		int_g_p[id] = (chi[id] - 1e-9)*(1-chi[id]) * df[id] * local_volume;
		int_g[id] = (chi[id] - 1e-9)*(1-chi[id]) * local_volume;


		__syncthreads();

		if ( id == 0 )
		{
			for ( int i = 1 ; i < numElements ; ++i )
				int_g_p[0] += int_g_p[i];
		}

		if ( id == 1 )
		{
			for ( int i = 1 ; i < numElements ; ++i )
				int_g[0] += int_g[i];
		}

		__syncthreads();

		if ( id == 0 )
			*p_w = int_g_p[0] / int_g[0];


	}
}

__global__ 
void calc_g_GPU(double*g, double* chi, size_t numElements, double local_volume)
{
	unsigned int id = threadIdx.x + blockIdx.x*blockDim.x;

	if (id < numElements)
	{
		g[id] = (chi[id] - 1e-9)*(1-chi[id]) * local_volume;

		// if ( id == 0 )
		// printf("%f\n", g[id]);
	}
}


__global__ 
void calc_g_GPU_(double* sum, double* chi, size_t numElements, double local_volume)
{
    int id = blockDim.x * blockIdx.x + threadIdx.x;
	int stride = blockDim.x*gridDim.x;
    
    // if ( id < numElements )
    // printf("%d : %e\n", id, x[id]);

	__shared__ double cache[1024];
    cache[threadIdx.x] = 0;
    
	double temp = 0.0;
	while(id < numElements)
	{
		temp += (chi[id] - 1e-9)*(1-chi[id]) * local_volume;
		id += stride;
	}
	
    cache[threadIdx.x] = temp;
    
	__syncthreads();
	
	// reduction
	unsigned int i = blockDim.x/2;
	while(i != 0){
		if(threadIdx.x < i){
			cache[threadIdx.x] += cache[threadIdx.x + i];
		}
		__syncthreads();
		i /= 2;
	}

	// reduce sum from all blocks' cache
	if(threadIdx.x == 0)
	{
		#if __CUDA_ARCH__ < 600
			atomicAdd_double(sum, cache[0]);
		#else
			atomicAdd(sum, cache[0]);
		#endif		
	}
		
}


// sum = sum ( df * g * local_volume)
__global__ 
void calcSum_df_g_GPU(double* sum, double* df, double* g, size_t numElements)
{
    int id = blockDim.x * blockIdx.x + threadIdx.x;
	int stride = blockDim.x*gridDim.x;
    
    // if ( id < n )
    // printf("%d : %e\n", id, x[id]);

	__shared__ double cache[1024];
    cache[threadIdx.x] = 0;
    
	double temp = 0.0;
	while(id < numElements)
	{
		temp += df[id]*g[id];	// local volume is already included in g, i.e. g = g*local_volume
		id += stride;
	}
	
    cache[threadIdx.x] = temp;
    
	__syncthreads();
	
	// reduction
	unsigned int i = blockDim.x/2;
	while(i != 0){
		if(threadIdx.x < i){
			cache[threadIdx.x] += cache[threadIdx.x + i];
		}
		__syncthreads();
		i /= 2;
	}

	// reduce sum from all blocks' cache
	if(threadIdx.x == 0)
	{
		#if __CUDA_ARCH__ < 600
			atomicAdd_double(sum, cache[0]);
		#else
			atomicAdd(sum, cache[0]);
		#endif		
	}
}




__host__
void calcP_w(double* p_w, double* sum_g, double* sum_df_g, double* df, double* chi, double* g, double* df_g, size_t numElements, double local_volume)
{	
	dim3 gridDim;
	dim3 blockDim;

	calculateDimensions(numElements, gridDim, blockDim);

	// calculate g of each element * local_volume
	calc_g_GPU<<<gridDim, blockDim>>>(g, chi, numElements, local_volume);
	
	// calculate sum_g = sum(g)
	sumOfVector_GPU<<<gridDim, blockDim>>>(sum_g, g, numElements);
	
	// sum_df_g = sum( g[i]*df[i]*local_volume )
	calcSum_df_g_GPU<<<gridDim, blockDim>>>(sum_df_g, df, g, numElements);

	// p_w = sum_df_g / sum_g
	divide_GPU<<<1,1>>>(p_w, sum_df_g, sum_g);

}



// sum = sum ( df * g * local_volume)
__global__ 
void calcSum_df_g_GPU_(double* sum, double* df, double* chi, size_t numElements, double local_volume)
{
    int id = blockDim.x * blockIdx.x + threadIdx.x;
	int stride = blockDim.x*gridDim.x;
    
    // if ( id < n )
    // printf("%d : %e\n", id, x[id]);

	__shared__ double cache[1024];
    cache[threadIdx.x] = 0;
    
	double temp = 0.0;
	while(id < numElements)
	{
		temp += df[id]* ( (chi[id] - 1e-9)*(1-chi[id]) * local_volume );
		id += stride;
	}
	
    cache[threadIdx.x] = temp;
    
	__syncthreads();
	
	// reduction
	unsigned int i = blockDim.x/2;
	while(i != 0){
		if(threadIdx.x < i){
			cache[threadIdx.x] += cache[threadIdx.x + i];
		}
		__syncthreads();
		i /= 2;
	}

	// reduce sum from all blocks' cache
	if(threadIdx.x == 0)
	{
		#if __CUDA_ARCH__ < 600
			atomicAdd_double(sum, cache[0]);
		#else
			atomicAdd(sum, cache[0]);
		#endif		
	}
}


__host__
void calcP_w_(double* p_w, double* sum_g, double* sum_df_g, double* df, double* chi, size_t numElements, double local_volume)
{	
	dim3 gridDim;
	dim3 blockDim;

	calculateDimensions(numElements, gridDim, blockDim);

	// calculate g of each element * local_volume
	// calculate sum_g = sum(g)
	calc_g_GPU_<<<gridDim, blockDim>>>(sum_g, chi, numElements, local_volume);

	// sum_df_g = sum( g[i]*df[i]*local_volume )
	calcSum_df_g_GPU_<<<gridDim, blockDim>>>(sum_df_g, df, chi, numElements, local_volume);

	// p_w = sum_df_g / sum_g
	divide_GPU<<<1,1>>>(p_w, sum_df_g, sum_g);


}

// two threads to calculate eta and beta
__global__ void calcEtaBeta( double* eta, double* beta, double etastar, double betastar, double* p_w )
{
	unsigned int id = threadIdx.x + blockIdx.x*blockDim.x;

	if ( id == 0 )	
		*eta = etastar * (*p_w);

	if ( id == 1 )
		*beta = betastar * (*p_w);

}


// convergence check in for the bisection algorithm in the density update process
__global__ void checkTDOConvergence(bool* foo, double rho, double* rho_trial)
{
	if ( abs(rho - *rho_trial) < 1e-7 )
		*foo = false;
	
	else
		*foo = true;
}


// computes and fills in the global stiffness matrix's ELL index array for 2d case
__global__ void fillIndexVector2D_GPU(size_t* index, size_t Nx, size_t Ny, size_t max_row_size, size_t num_rows)
{
	unsigned int id = threadIdx.x + blockIdx.x*blockDim.x;

	int counter = 0;
	int dim = 2;

	if ( id < num_rows )
	{

		int base_id = (id - id%dim);
		
		// south-west
		if ( id  >= (Nx + 1)*dim && (id) % ((Nx + 1)*dim) >= dim )
		{
			for(int i = 0 ; i < dim ; i++)
			{
				index[id + counter*num_rows] = (id - id%dim) - (Nx+1)*dim - dim + i;
				counter++;
			}
		}

		// south
		if ( id  >= (Nx + 1)*dim )
		{
			for(int i = 0 ; i < dim ; i++)
			{
				index[id + counter*num_rows] = (id - id%dim) - (Nx+1)*dim + i;
				counter++;
			}
		}

		// south-east
		if ( id  >= (Nx + 1)*dim && (base_id) % ((Nx*dim) + (base_id/(2*(Nx+1)))*dim*(Nx+1)) != 0 )
		{
			for(int i = 0 ; i < dim ; i++)
			{
				index[id + counter*num_rows] = (id - id%dim) - (Nx+1)*dim + dim + i;
				counter++;
			}
		}

		// west
		if ( (id) % ((Nx + 1)*dim) >= dim )
		{
			for(int i = 0 ; i < dim ; i++)
			{
				index[id + counter*num_rows] = (id - id%dim) - dim + i;
				counter++;
			}
		}

		// origin
		for(int i = 0 ; i < dim ; i++)
			{
				index[id + counter*num_rows] = (id - id%dim) + i;
				counter++;
			}

		// east
		if ( base_id == 0 || (base_id) % ((Nx*dim) + (base_id/(2*(Nx+1)))*dim*(Nx+1)) != 0 )
		{
			for(int i = 0 ; i < dim ; i++)
			{
				index[id + counter*num_rows] = (id - id%dim) + dim + i;
				counter++;
			}
		}

		// north-west
		if ( id < (Nx+1)*(Ny)*dim && (id) % ((Nx + 1)*dim) >= dim )
		{
			for(int i = 0 ; i < dim ; i++)
			{
				index[id + counter*num_rows] = (id - id%dim) + (Nx+1)*dim - dim + i;
				counter++;
			}
		}

		// north
		if ( id < (Nx+1)*(Ny)*dim )
		{
			for(int i = 0 ; i < dim ; i++)
			{
				index[id + counter*num_rows] = (id - id%dim) + (Nx+1)*dim + i;
				counter++;
			}
		}

		// north-east
		if ( base_id == 0 || id < (Nx+1)*(Ny)*dim && (base_id) % ((Nx*dim) + (base_id/(2*(Nx+1)))*dim*(Nx+1)) != 0 )
		{
			for(int i = 0 ; i < dim ; i++)
			{
				index[id + counter*num_rows] = (id - id%dim) + (Nx+1)*dim + dim + i;
				counter++;
			}
		}

		for ( int i = counter ; i < max_row_size; i++)
		{
			index[id + i*num_rows] = num_rows;
		}

	}	
}

// computes and fills in the global stiffness matrix's ELL index array for 3d case
__global__ void fillIndexVector3D_GPU(size_t* index, size_t Nx, size_t Ny, size_t Nz, size_t max_row_size, size_t num_rows)
{
	unsigned int id = threadIdx.x + blockIdx.x*blockDim.x;

	int counter = 0;
	int dim = 3;

	if ( id < num_rows )
	{
	
		size_t base_id = (id - id%dim);
		size_t gridsize_2D = (Nx+1)*(Ny+1)*dim;

		// boolean variables that returns true if the neighbouring node exists
		bool prev_layer = (id >= (Nx+1)*(Ny+1)*dim);
		bool next_layer = (id < (Nx+1)*(Ny+1)*(Nz)*dim);
		bool south = ((id) % ((Nx + 1)*(Ny + 1)*dim)  >= (Nx + 1)*dim);
		bool north = ((id) % ((Nx + 1)*(Ny + 1)*dim) < (Nx+1)*(Ny)*dim);
		bool west = ((id) % ((Nx + 1)*dim) >= dim);
		bool east = ((base_id) % ((Nx*dim) + (base_id/(3*(Nx+1)))*dim*(Nx+1)) != 0);
	

	//// previous layer
		
		// south-west
		if ( prev_layer && south && west )
		{
			for(int i = 0 ; i < dim ; i++)
			{
				index[id + counter*num_rows] = (id - id%dim) - (Nx+1)*dim - dim + i - gridsize_2D;
				counter++;
			}
		}

		// south
		if ( prev_layer && south )
		{
			for(int i = 0 ; i < dim ; i++)
			{
				index[id + counter*num_rows] = (id - id%dim) - (Nx+1)*dim + i - gridsize_2D;
				counter++;
			}
		}

		// south-east
		if ( prev_layer && south && east )
		{	
			for(int i = 0 ; i < dim ; i++)
			{
				index[id + counter*num_rows] = (id - id%dim) - (Nx+1)*dim + dim + i - gridsize_2D;
				counter++;
			}
		}

		// west
		if ( prev_layer && west )
		{
			for(int i = 0 ; i < dim ; i++)
			{
				index[id + counter*num_rows] = (id - id%dim) - dim + i - gridsize_2D;
				counter++;
			}
		}

		// origin
		if ( prev_layer )
		{
			for(int i = 0 ; i < dim ; i++)
			{
				index[id + counter*num_rows] = (id - id%dim) + i - gridsize_2D;
				counter++;
			}
		}



		// east
		if ( prev_layer && east )
		{

			for(int i = 0 ; i < dim ; i++)
				{
					index[id + counter*num_rows] = (id - id%dim) + dim + i - gridsize_2D;
					counter++;
				}
		}

		// north-west
		if ( prev_layer && north && west )
		{
			for(int i = 0 ; i < dim ; i++)
			{
				index[id + counter*num_rows] = (id - id%dim) + (Nx+1)*dim - dim + i - gridsize_2D;
				counter++;
			}
		}

		// north
		if ( prev_layer && north )
		{
			for(int i = 0 ; i < dim ; i++)
			{
				index[id + counter*num_rows] = (id - id%dim) + (Nx+1)*dim + i - gridsize_2D;
				counter++;
			}

		}

		// north-east
		if ( prev_layer && north && east )
		{
			for(int i = 0 ; i < dim ; i++)
			{
				index[id + counter*num_rows] = (id - id%dim) + (Nx+1)*dim + dim + i - gridsize_2D;
				counter++;
			}
		}

	//// current layer
		
		// south-west
		if ( south && west )
		{
			for(int i = 0 ; i < dim ; i++)
			{
				index[id + counter*num_rows] = (id - id%dim) - (Nx+1)*dim - dim + i;
				counter++;
			}
		}

		// south
		if ( south )
		{
			for(int i = 0 ; i < dim ; i++)
			{
				index[id + counter*num_rows] = (id - id%dim) - (Nx+1)*dim + i;
				counter++;
			}
		}

		// south-east
		if ( south && east )
		{	
			for(int i = 0 ; i < dim ; i++)
			{
				index[id + counter*num_rows] = (id - id%dim) - (Nx+1)*dim + dim + i;
				counter++;
			}
		}

		// west
		if ( west )
		{
			for(int i = 0 ; i < dim ; i++)
			{
				index[id + counter*num_rows] = (id - id%dim) - dim + i;
				counter++;
			}
		}

		// origin
		for(int i = 0 ; i < dim ; i++)
		{
			index[id + counter*num_rows] = (id - id%dim) + i;
			counter++;
		}

		// east
		if ( base_id == 0 || east )
		{

			for(int i = 0 ; i < dim ; i++)
				{
					index[id + counter*num_rows] = (id - id%dim) + dim + i;
					counter++;
				}
		}

		// north-west
		if ( north && west )
		{
			for(int i = 0 ; i < dim ; i++)
			{
				index[id + counter*num_rows] = (id - id%dim) + (Nx+1)*dim - dim + i;
				counter++;
			}

		}

		// north
		if ( north )
		{
			for(int i = 0 ; i < dim ; i++)
			{
				index[id + counter*num_rows] = (id - id%dim) + (Nx+1)*dim + i;
				counter++;
			}

		}

		// north-east
		if ( base_id == 0 || (north && east ) )
		{
			for(int i = 0 ; i < dim ; i++)
			{
				index[id + counter*num_rows] = (id - id%dim) + (Nx+1)*dim + dim + i;
				counter++;
			}
		}


	//// next layer
	
		// south-west
		if ( next_layer && south && west )
		{
			for(int i = 0 ; i < dim ; i++)
			{
				index[id + counter*num_rows] = (id - id%dim) - (Nx+1)*dim - dim + i + gridsize_2D;
				counter++;
			}
		}

		// south
		if ( next_layer && south )
		{
			for(int i = 0 ; i < dim ; i++)
			{
				index[id + counter*num_rows] = (id - id%dim) - (Nx+1)*dim + i + gridsize_2D;
				counter++;
			}

		}

		// south-east
		if ( next_layer && south && east )
		{	
			for(int i = 0 ; i < dim ; i++)
			{
				index[id + counter*num_rows] = (id - id%dim) - (Nx+1)*dim + dim + i + gridsize_2D;
				counter++;
			}

		}

		// west
		if ( next_layer && west )
		{

			for(int i = 0 ; i < dim ; i++)
			{
				index[id + counter*num_rows] = (id - id%dim) - dim + i + gridsize_2D;
				counter++;
			}

		}

		// origin
		if ( next_layer )
		{
			for(int i = 0 ; i < dim ; i++)
			{
				index[id + counter*num_rows] = (id - id%dim) + i + gridsize_2D;
				counter++;
			}
		}

		// east
		if ( base_id == 0 || ( next_layer && east ) )
		{

			for(int i = 0 ; i < dim ; i++)
				{
					index[id + counter*num_rows] = (id - id%dim) + dim + i + gridsize_2D;
					counter++;
				}
		}

		// north-west
		if ( next_layer && north && west )
		{
			for(int i = 0 ; i < dim ; i++)
			{
				index[id + counter*num_rows] = (id - id%dim) + (Nx+1)*dim - dim + i + gridsize_2D;
				counter++;
			}
		}

		// north
		if ( next_layer && north )
		{
			for(int i = 0 ; i < dim ; i++)
			{
				index[id + counter*num_rows] = (id - id%dim) + (Nx+1)*dim + i + gridsize_2D;
				counter++;
			}
		}

		// north-east
		if ( base_id == 0 || (next_layer && north && east ) )
		{
			for(int i = 0 ; i < dim ; i++)
			{
				index[id + counter*num_rows] = (id - id%dim) + (Nx+1)*dim + dim + i + gridsize_2D;
				counter++;
			}
		}

		for ( int i = counter ; i < max_row_size; i++)
		{
			index[id + i*num_rows] = num_rows;
		}

	}

}

// assembles the prolongation matrix for 2d case
// the ELL value and index arrays are calculated and filled
__global__ void fillProlMatrix2D_GPU(double* p_value, size_t* p_index, size_t Nx, size_t Ny, size_t p_max_row_size, size_t num_rows, size_t num_cols)
{
	unsigned int id = threadIdx.x + blockIdx.x*blockDim.x;

	if ( id < num_rows )
	{
		int counter = 0;
		int dim = 2;	

		// coarse grid
		size_t Nx_ = Nx / 2;
		size_t Ny_ = Ny / 2;

		size_t base_id = (id - id%dim);
		size_t node_index = base_id / dim;
		int coarse_node_index = getCoarseNode_GPU(node_index, Nx, Ny, 0, dim);
		
		// if node is even numbered
		bool condition1 = (node_index % 2 == 0 );

		// if node exists in the coarse grid
		bool condition2 = ( node_index % ((Nx+1)*2) < (Nx + 1) );

		bool south = ( id  >= (Nx + 1)*dim );
		bool west  = ( (id) % ((Nx + 1)*dim) >= dim );
		bool east  = ( (base_id) % ((Nx*dim) + (base_id/(2*(Nx+1)))*dim*(Nx+1)) != 0 );
		bool north = ( id < (Nx+1)*(Ny)*dim );

		// if there exists a coarse node in the same location
		if ( getFineNode_GPU(coarse_node_index, Nx_, Ny_, 0, dim) == node_index )
		{
			p_index[id + counter*num_rows] = coarse_node_index*dim + id%dim;
			p_value[id + counter*num_rows] = 1;
			counter++;
		}

		else
		{
			// south-west
			if ( south && condition1 && !condition2 && west ) 
			{
				size_t south_west_fine_node = (node_index - (Nx+1) - 1);
				size_t south_west_coarse_node = getCoarseNode_GPU(south_west_fine_node, Nx, Ny, 0, dim);
				p_index[id + counter*num_rows] = south_west_coarse_node*dim + id%dim ;
				p_value[id + counter*num_rows] = 0.25 ;
				counter++;
			}

			// south
			if ( south && !condition1 && !condition2 )
			{
				size_t south_fine_node = (node_index - (Nx+1) );
				size_t south_coarse_node = getCoarseNode_GPU(south_fine_node, Nx, Ny, 0, dim);
				p_index[id + counter*num_rows] = south_coarse_node*dim + id%dim ;
				p_value[id + counter*num_rows] = 0.5 ;
				counter++;
			}

			// south-east
			if ( south && condition1 && !condition2 && east ) 
			{
				size_t south_east_fine_node = (node_index - (Nx+1) + 1);
				size_t south_east_coarse_node = getCoarseNode_GPU(south_east_fine_node, Nx, Ny, 0, dim);
				p_index[id + counter*num_rows] = south_east_coarse_node*dim + id%dim ;
				p_value[id + counter*num_rows] = 0.25 ;
				counter++;
			}

			// west
			if ( west && condition2 )
			{
				size_t west_fine_node = (node_index - 1);
				size_t west_coarse_node = getCoarseNode_GPU(west_fine_node, Nx, Ny, 0, dim);
				p_index[id + counter*num_rows] = west_coarse_node*dim + id%dim ;
				p_value[id + counter*num_rows] = 0.5 ;
				counter++;
			}

			// east
			if ( east && condition2 )
			{
				size_t east_fine_node = (node_index + 1);
				size_t east_coarse_node = getCoarseNode_GPU(east_fine_node, Nx, Ny, 0, dim);
				p_index[id + counter*num_rows] = east_coarse_node*dim + id%dim ;
				p_value[id + counter*num_rows] = 0.5 ;
				counter++;
			}

			// north-west
			if ( north && condition1 && !condition2 && west )
			{
				size_t north_west_fine_node = (node_index + (Nx+1) - 1);
				size_t north_west_coarse_node = getCoarseNode_GPU(north_west_fine_node, Nx, Ny, 0, dim);
				p_index[id + counter*num_rows] = north_west_coarse_node*dim + id%dim ;
				p_value[id + counter*num_rows] = 0.25 ;
				counter++;
			}

			// north
			if ( north && !condition1 && !condition2 )
			{
				size_t north_fine_node = (node_index + (Nx+1) );
				size_t north_coarse_node = getCoarseNode_GPU(north_fine_node, Nx, Ny, 0, dim);
				p_index[id + counter*num_rows] = north_coarse_node*dim + id%dim ;
				p_value[id + counter*num_rows] = 0.5 ;
				counter++;
			}

			// north-east
			if ( north && condition1 && !condition2 && east ) 
			{
				size_t north_east_fine_node = (node_index + (Nx+1) + 1);
				size_t north_east_coarse_node = getCoarseNode_GPU(north_east_fine_node, Nx, Ny, 0, dim);
				p_index[id + counter*num_rows] = north_east_coarse_node*dim + id%dim ;
				p_value[id + counter*num_rows] = 0.25 ;
				counter++;
			}

		}
		
		// remaining entries are filled with num_cols
		for ( int i = counter ; i < p_max_row_size; i++)
		{
			p_index[id + i*num_rows] = num_cols;
		}

	}
}

// assembles the prolongation matrix for 3d case
// the ELL value and index arrays are calculated and filled
__global__ void fillProlMatrix3D_GPU(double* p_value, size_t* p_index, size_t Nx, size_t Ny, size_t Nz, size_t p_max_row_size, size_t num_rows, size_t num_cols)
{
	unsigned int id = threadIdx.x + blockIdx.x*blockDim.x;

	if ( id < num_rows )
	{
		int counter = 0;
		int dim = 3;	

		// coarse grid
		size_t Nx_ = Nx / 2;
		size_t Ny_ = Ny / 2;
		size_t Nz_ = Nz / 2;

		size_t base_id = (id - id%dim);
		size_t id_2D = (id) % ((Nx+1)*(Ny+1)*dim);
		size_t node_index = base_id / dim;
		int coarse_node_index = getCoarseNode3D_GPU(node_index, Nx, Ny, Nz);

		size_t numNodes2D = (Nx+1)*(Ny+1);


		// if node is even numbered
		bool condition1 = ( node_index % 2 == 0 );

		
		bool condition5 = ( (id_2D/dim) % ((Nx+1)*2) < (Nx+1) );
		bool condition6 = ( node_index % (numNodes2D*2) < (Nx+1)*(Ny+1) );

		// if there exists a coarse node in the same location
		if ( getFineNode_GPU(coarse_node_index, Nx_, Ny_, Nz_, dim) == node_index )
		{
			p_index[id + counter*num_rows] = coarse_node_index*dim + id%dim;
			p_value[id + counter*num_rows] = 1;
			counter++;
		}

		// diagonals
		else if ( !condition1 && !condition5 && !condition6 )
		{
			size_t fine_node;
			size_t coarse_node;

			// previous-south-west
			fine_node = (node_index - numNodes2D - (Nx+1) - 1 );
			coarse_node = getCoarseNode3D_GPU(fine_node, Nx, Ny, Nz);
			p_index[id + counter*num_rows] = coarse_node*dim + id%dim ;
			p_value[id + counter*num_rows] = 0.125 ;
			counter++;

			// previous-south-east
			fine_node = (node_index - numNodes2D - (Nx+1) + 1 );
			coarse_node = getCoarseNode3D_GPU(fine_node, Nx, Ny, Nz);
			p_index[id + counter*num_rows] = coarse_node*dim + id%dim ;
			p_value[id + counter*num_rows] = 0.125 ;
			counter++;

			// previous-north-west
			fine_node = (node_index - numNodes2D + (Nx+1) - 1 );
			coarse_node = getCoarseNode3D_GPU(fine_node, Nx, Ny, Nz);
			p_index[id + counter*num_rows] = coarse_node*dim + id%dim ;
			p_value[id + counter*num_rows] = 0.125 ;
			counter++; 

			// previous-north-east
			fine_node = (node_index - numNodes2D + (Nx+1) + 1 );
			coarse_node = getCoarseNode3D_GPU(fine_node, Nx, Ny, Nz);
			p_index[id + counter*num_rows] = coarse_node*dim + id%dim ;
			p_value[id + counter*num_rows] = 0.125 ;
			counter++; 

			// next-south-west
			fine_node = (node_index + numNodes2D - (Nx+1) - 1 );
			coarse_node = getCoarseNode3D_GPU(fine_node, Nx, Ny, Nz);
			p_index[id + counter*num_rows] = coarse_node*dim + id%dim ;
			p_value[id + counter*num_rows] = 0.125 ;
			counter++;

			// next-south-east
			fine_node = (node_index + numNodes2D - (Nx+1) + 1);
			coarse_node = getCoarseNode3D_GPU(fine_node, Nx, Ny, Nz);
			p_index[id + counter*num_rows] = coarse_node*dim + id%dim ;
			p_value[id + counter*num_rows] = 0.125 ;
			counter++;

			// next-north-west
			fine_node = (node_index + numNodes2D + (Nx+1) - 1 );
			coarse_node = getCoarseNode3D_GPU(fine_node, Nx, Ny, Nz);
			p_index[id + counter*num_rows] = coarse_node*dim + id%dim ;
			p_value[id + counter*num_rows] = 0.125 ;
			counter++; 

			// next-north-east
			fine_node = (node_index + numNodes2D + (Nx+1) + 1);
			coarse_node = getCoarseNode3D_GPU(fine_node, Nx, Ny, Nz);
			p_index[id + counter*num_rows] = coarse_node*dim + id%dim ;
			p_value[id + counter*num_rows] = 0.125 ;
			counter++; 
		}

		// diagonals on x-z plane
		else if ( condition1 && condition5 && !condition6 )
		{
			size_t fine_node;
			size_t coarse_node;

			// previous-west
			fine_node = (node_index - (Nx+1)*(Ny+1) - 1);
			coarse_node = getCoarseNode3D_GPU(fine_node, Nx, Ny, Nz);
			p_index[id + counter*num_rows] = coarse_node*dim + id%dim ;
			p_value[id + counter*num_rows] = 0.25 ;
			counter++;

			// previous-east
			fine_node = (node_index - (Nx+1)*(Ny+1) + 1);
			coarse_node = getCoarseNode3D_GPU(fine_node, Nx, Ny, Nz);
			p_index[id + counter*num_rows] = coarse_node*dim + id%dim ;
			p_value[id + counter*num_rows] = 0.25 ;
			counter++;

			// next-west
			fine_node = (node_index + (Nx+1)*(Ny+1) - 1);
			coarse_node = getCoarseNode3D_GPU(fine_node, Nx, Ny, Nz);
			p_index[id + counter*num_rows] = coarse_node*dim + id%dim ;
			p_value[id + counter*num_rows] = 0.25 ;
			counter++;

			// next-east
			fine_node = (node_index + (Nx+1)*(Ny+1) + 1);
			coarse_node = getCoarseNode3D_GPU(fine_node, Nx, Ny, Nz);
			p_index[id + counter*num_rows] = coarse_node*dim + id%dim ;
			p_value[id + counter*num_rows] = 0.25 ;
			counter++;
		}

		// diagonals in x-y plane
		else if ( condition1 && !condition5 && condition6 )
		{
			size_t fine_node;
			size_t coarse_node;

			// south-west
			fine_node = (node_index - (Nx+1) - 1);
			coarse_node = getCoarseNode3D_GPU(fine_node, Nx, Ny, Nz);
			p_index[id + counter*num_rows] = coarse_node*dim + id%dim ;
			p_value[id + counter*num_rows] = 0.25 ;
			counter++;

			// south-east
			fine_node = (node_index - (Nx+1) + 1);
			coarse_node = getCoarseNode3D_GPU(fine_node, Nx, Ny, Nz);
			p_index[id + counter*num_rows] = coarse_node*dim + id%dim ;
			p_value[id + counter*num_rows] = 0.25 ;
			counter++;

			// north-east
			fine_node = (node_index + (Nx+1) - 1);
			coarse_node = getCoarseNode3D_GPU(fine_node, Nx, Ny, Nz);
			p_index[id + counter*num_rows] = coarse_node*dim + id%dim ;
			p_value[id + counter*num_rows] = 0.25 ;
			counter++;

			// north-east
			fine_node = (node_index + (Nx+1) + 1);
			coarse_node = getCoarseNode3D_GPU(fine_node, Nx, Ny, Nz);
			p_index[id + counter*num_rows] = coarse_node*dim + id%dim ;
			p_value[id + counter*num_rows] = 0.25 ;
			counter++;
		}

		// diagonals in y-z plane
		else if ( condition1 && !condition5 && !condition6 )
		{
			size_t fine_node;
			size_t coarse_node;

			// previous-south
			fine_node = (node_index - (Nx+1)*(Ny+1) - (Nx+1) );
			coarse_node = getCoarseNode3D_GPU(fine_node, Nx, Ny, Nz);
			p_index[id + counter*num_rows] = coarse_node*dim + id%dim ;
			p_value[id + counter*num_rows] = 0.25 ;
			counter++;

			// previous-north
			fine_node = (node_index - (Nx+1)*(Ny+1) + (Nx+1) );
			coarse_node = getCoarseNode3D_GPU(fine_node, Nx, Ny, Nz);
			p_index[id + counter*num_rows] = coarse_node*dim + id%dim ;
			p_value[id + counter*num_rows] = 0.25 ;
			counter++;

			// next-south
			fine_node = (node_index + (Nx+1)*(Ny+1) - (Nx+1) );
			coarse_node = getCoarseNode3D_GPU(fine_node, Nx, Ny, Nz);
			p_index[id + counter*num_rows] = coarse_node*dim + id%dim ;
			p_value[id + counter*num_rows] = 0.25 ;
			counter++;

			// next-north
			fine_node = (node_index + (Nx+1)*(Ny+1) + (Nx+1) );
			coarse_node = getCoarseNode3D_GPU(fine_node, Nx, Ny, Nz);
			p_index[id + counter*num_rows] = coarse_node*dim + id%dim ;
			p_value[id + counter*num_rows] = 0.25 ;
			counter++;
		}

		else
		{		
			// previous-origin
			if ( !condition1 && condition5 && !condition6 )
			{
				// printf("%lu\n", node_index*dim );
				size_t fine_node = (node_index - (Nx+1)*(Ny+1));
				size_t coarse_node = getCoarseNode3D_GPU(fine_node, Nx, Ny, Nz);
				p_index[id + counter*num_rows] = coarse_node*dim + id%dim ;
				p_value[id + counter*num_rows] = 0.5 ;
				counter++;
			}

			// next-origin
			if ( !condition1 && condition5 && !condition6 )
			{
				// printf("%lu\n", node_index*dim );
				size_t fine_node = (node_index + (Nx+1)*(Ny+1));
				size_t coarse_node = getCoarseNode3D_GPU(fine_node, Nx, Ny, Nz);
				p_index[id + counter*num_rows] = coarse_node*dim + id%dim ;
				p_value[id + counter*num_rows] = 0.5 ;
				counter++;
			}

			// south
			if ( !condition1 && !condition5 && condition6 )
			{
				
				size_t fine_node = (node_index - (Nx+1));
				size_t coarse_node = getCoarseNode3D_GPU(fine_node, Nx, Ny, Nz);
				p_index[id + counter*num_rows] = coarse_node*dim + id%dim ;
				p_value[id + counter*num_rows] = 0.5 ;
				counter++;
			}

			// west
			if ( !condition1 && condition5 && condition6 )
			{
				// printf("%lu\n", node_index*3 );
				size_t fine_node = (node_index - 1);
				size_t coarse_node = getCoarseNode3D_GPU(fine_node, Nx, Ny, Nz);
				p_index[id + counter*num_rows] = coarse_node*dim + id%dim ;
				p_value[id + counter*num_rows] = 0.5 ;
				counter++;
			}

			// east
			if ( !condition1 && condition5 && condition6 )
			{
				
				size_t fine_node = (node_index + 1);
				size_t coarse_node = getCoarseNode3D_GPU(fine_node, Nx, Ny, Nz);
				p_index[id + counter*num_rows] = coarse_node*dim + id%dim ;
				p_value[id + counter*num_rows] = 0.5 ;
				counter++;
			}

			// north
			if ( !condition1 && !condition5 && condition6 )
			{
				
				size_t fine_node = (node_index + (Nx+1));
				size_t coarse_node = getCoarseNode3D_GPU(fine_node, Nx, Ny, Nz);
				p_index[id + counter*num_rows] = coarse_node*dim + id%dim ;
				p_value[id + counter*num_rows] = 0.5 ;
				counter++;
			}

		}

		for ( int i = counter ; i < p_max_row_size; i++)
			{
				p_index[id + i*num_rows] = num_cols;
			}

	}
}

// obtaining a node's corresponding node on a coarser grid
__device__ int getCoarseNode_GPU(size_t index, size_t Nx, size_t Ny, size_t Nz, size_t dim)
{
	// get coarse grid dimensions
	size_t Nx_ = Nx / 2;
	// size_t Ny_ = Ny / 2;
	// size_t Nz_ = Nz / 2;

	// if node is even numbered
	bool condition1 = (index % 2 == 0 );

	// if node exists in the coarse grid
	bool condition2 = ( index % ((Nx+1)*2) < (Nx + 1) );

	if ( condition1 && condition2 )
	{
		return index/2 - (index/((Nx+1)*2 ))*(Nx_);
	}

	// -1 means the node in the coarse grid does not exist
	else
		return -1;
}


__device__ int getCoarseNode3D_GPU(size_t index, size_t Nx, size_t Ny, size_t Nz)
{
	// get coarse grid dimensions
	size_t Nx_ = Nx / 2;
	size_t Ny_ = Ny / 2;
	// size_t Nz_ = Nz / 2;

	size_t gridsize2D = (Nx+1)*(Ny+1);
	size_t gridsize2D_ = (Nx_+1)*(Ny_+1);

	// if node is even numbered
	bool condition1 = ( index % 2 == 0 );

	// if node exists in the coarse grid (x-y-plane)
	bool condition2 = ( index % ((Nx+1)*2) < (Nx + 1) );

	// if node exists in the coarse grid (y-z-plane)
	bool condition3 = ( index % ((Nx+1)*(Ny+1)*2) < (Nx+1)*(Ny+1) );

	if ( condition1 && condition2 && condition3 )
	{
		int base_id = index % gridsize2D;

		return base_id/2 - (base_id/((Nx+1)*2 ))*(Nx_) + (index/(gridsize2D*2))*gridsize2D_;
		// return index/2 - (index/((Nx+1)*2 ))*(Nx_);
	}

	// -1 means the node in the coarse grid does not exist
	else
		return -1;
}





// DEBUG: check to ensure mass is conserved during the density update process
__global__ void checkMassConservation(double* chi, double local_volume, size_t numElements)
{
    unsigned int id = threadIdx.x + blockIdx.x*blockDim.x;

    __shared__ double temp[1024];
    
    if ( id < numElements)
    {
        // sum of chi * local_volume
        temp[id] = chi[id] * local_volume;
    }
	__syncthreads();

    if ( id == 0 )
    {
        for ( int i = 1 ; i < numElements ; i++ )
        {
            temp[0] += temp[i];
        }

		// total volume
		double vol = local_volume * numElements;

        printf("chi_trial %f\n", temp[0] / vol);
    }
}



// adds the value to a transposed ELLPack matrix A at (row,col)
__device__
void atomicAddAt( size_t row, size_t col, double* vValue, size_t* vIndex, size_t max_row_size, size_t num_rows, double value )
{
    for(size_t k = 0; k < max_row_size; ++k)
    {	
        if(vIndex[k * num_rows + col] == row)
        {
            atomicAdd( &vValue[k * num_rows + col] , value );
                k = max_row_size; // to exit for loop
		}
    }
}



// A_coarse = P^T * A_fine * P
// A : fine stiffness matrix
// A_ : coarse stiffness matrix
// P : prolongation matrix
__global__ void PTAP(double* A_value, size_t* A_index, size_t max_row_size, size_t num_rows,
	double* A_value_, size_t* A_index_, size_t max_row_size_, size_t num_rows_,
	double* P_value, size_t* P_index, size_t p_max_row_size)
{
	int k = blockDim.x * blockIdx.x + threadIdx.x;

	if( k < num_rows )
	{
		for ( int i_ = 0 ; i_ < p_max_row_size ; i_++ )
		{
			size_t i = P_index[k + i_*num_rows];
			double P_ki = P_value[k + i_*num_rows];

			for( int l_ = 0 ; l_ < max_row_size ; l_++  )
			{
				size_t l = A_index[k + l_*num_rows];
				double A_kl = A_value[k + l_*num_rows];
				double P_ki_A_kl = P_ki * A_kl;

				for( int j_ = 0 ; j_ < p_max_row_size ; j_++ )
				{
					size_t j = P_index[l + j_*num_rows];
					
					double P_lj = P_value[l + j_*num_rows];
					double P_ki_A_kl_P_lj = P_ki_A_kl * P_lj;
									
					if(P_ki_A_kl_P_lj != 0.0)
						atomicAddAt( j, i, A_value_, A_index_, max_row_size_, num_rows_, P_ki_A_kl_P_lj );
				}
			}
		}
	}
}

// calculation of compliance, c = 0.5 * sum( u^T * K * u )
// c is labelled as sum
__global__ 
void calcCompliance(double* sum, double* u, double* chi, size_t* node_index, double* d_A_local, double local_volume, size_t num_rows, size_t dim, size_t numElements)
{
	int id = blockDim.x * blockIdx.x + threadIdx.x;

	if ( id < numElements)
	{
		double uTKu = 0;
		double temp[24];
		size_t numNodesPerElement = pow(2,dim);
		
		uTKu = 0;
		for ( int n = 0; n < num_rows; n++ )
		{
			temp[n]=0;
			for ( int m = 0; m < num_rows; m++)
			{
				// converts local node to global node
				int global_col = ( node_index [ (m / dim) + id*numNodesPerElement ] * dim ) + ( m % dim ); 
				temp[n] += u[global_col] * d_A_local[ n + m*num_rows ];
			}

		}
		
		for ( int n = 0; n < num_rows; n++ )
		{
			int global_col = ( node_index [ (n / dim) + id*numNodesPerElement ] * dim ) + ( n % dim );
			uTKu += temp[n] * u[global_col];
		}
		
		__syncthreads();
		uTKu *= 0.5 * pow(chi[id],3);
		
		// reduction
		__shared__ double cache[1024];
		cache[threadIdx.x] = uTKu;
				
		__syncthreads();
		
		// reduction
		unsigned int i = blockDim.x/2;
		while(i != 0){
			if(threadIdx.x < i){
				cache[threadIdx.x] += cache[threadIdx.x + i];
			}
			__syncthreads();
			i /= 2;
		}

		// reduce sum from all blocks' cache
		if(threadIdx.x == 0)
		{
			#if __CUDA_ARCH__ < 600
				atomicAdd_double(sum, cache[0]);
			#else
				atomicAdd(sum, cache[0]);
			#endif		
		}	
	}
		
}

// computes the measure of non-discreteness (MOD)
__global__ 
void calcMOD(double* sum, double* chi, double local_volume, size_t numElements)
{
    int id = blockDim.x * blockIdx.x + threadIdx.x;
	int stride = blockDim.x*gridDim.x;
    
	__shared__ double cache[1024];
    cache[threadIdx.x] = 0;
	

	double temp = 0.0;
	while(id < numElements)
	{
		temp += chi[id] * (1-chi[id]) * local_volume * 4 / ( local_volume * numElements );
		id += stride;
	}
	
    cache[threadIdx.x] = temp;
    
	__syncthreads();
	
	// reduction
	unsigned int i = blockDim.x/2;
	while(i != 0){
		if(threadIdx.x < i){
			cache[threadIdx.x] += cache[threadIdx.x + i];
		}
		__syncthreads();
		i /= 2;
	}

	// reduce sum from all blocks' cache
	if(threadIdx.x == 0)
	{
		#if __CUDA_ARCH__ < 600
			atomicAdd_double(sum, cache[0]);
		#else
			atomicAdd(sum, cache[0]);
		#endif		
	}
		
}