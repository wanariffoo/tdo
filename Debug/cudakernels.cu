
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include <ctime>
#include <iostream>
#include "cudakernels.h"


#define CUDA_CALL( call )                                                                                          \
    {                                                                                                                  \
    cudaError_t err = call;                                                                                          \
    if ( cudaSuccess != err)                                                                                         \
        fprintf(stderr, "CUDA error for %s in %d of %s : %s.\n", #call , __LINE__ , __FILE__ ,cudaGetErrorString(err));\
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


// Determines 1-dimensional CUDA block and grid sizes based on the number of rows N
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

// Determines 2-dimensional CUDA block and grid sizes based on the number of rows N
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

// TODO: this is for 2D only, need 3D later
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

// adds the value to an ELLPack matrix A at (x,y)
__device__
void addAt( size_t x, size_t y, double* vValue, size_t* vIndex, size_t max_row_size, double value )
{
    for(size_t k = 0; k < max_row_size; ++k)
    {
        if(vIndex[y * max_row_size + k] == x)
        {
            
            vValue[y * max_row_size + k] += value;
            // printf("%f \n", vValue[x * max_row_size + k]);
                k = max_row_size; // to exit for loop
            }
    }
}

// sets the value of an ELLPack matrix A at (x,y)
__device__
void setAt( size_t x, size_t y, double* vValue, size_t* vIndex, size_t max_row_size, double value )
{
    for(size_t k = 0; k < max_row_size; ++k)
    {
        if(vIndex[y * max_row_size + k] == x)
        {
            vValue[y * max_row_size + k] = value;
                k = max_row_size; // to exit for loop
            }
    }
}

__global__
void setToZero(double* a, size_t num_rows)
{
	int id = blockDim.x * blockIdx.x + threadIdx.x;

	if ( id < num_rows )
		a[id] = 0.0;
}

// norm = x.norm()
__global__ 
void norm_GPU(double* norm, double* x, size_t num_rows)
{
	int id = blockDim.x * blockIdx.x + threadIdx.x;

	// TODO: if (id < num)

	if ( id == 0 )
		*norm = 0;
	__syncthreads();

	if ( id < num_rows )
	{
		atomicAdd_double( norm, x[id]*x[id] );
	}
	__syncthreads();

	if ( id == 0 )
		*norm = sqrt(*norm);
}

// a[] = 0


// a[] = 0, size_t
__global__
void setToZero(size_t* a, size_t num_rows)
{
	int id = blockDim.x * blockIdx.x + threadIdx.x;

	if ( id < num_rows )
		a[id] = 0.0;
}

//TODO: to delete
// bool = true
__global__
void setToTrue( bool *foo )
{
	*foo = true;
}


// DEBUG: TEST !!!!!!!!!!!!!!!!!!!!!!!!!!
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
		atomicAdd_double(sum, cache[0]);
}


__global__ 
void LastBlockSumOfSquare_GPU(double* sum, double* x, size_t n, size_t counter)
{
	int id = threadIdx.x + blockDim.x*blockIdx.x;
    
    // if ( id >= counter*blockDim.x && id < ( ( counter*blockDim.x ) + lastBlockSize ) )
    if ( id >= counter*blockDim.x && id < n )
		atomicAdd_double(sum, x[id]*x[id]);
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
    // sumOfSquare_GPU<<<gridDim.x - 1, blockDim>>>(d_norm, d_x, N); // TODO: check, this is the original
    sumOfSquare_GPU<<<gridDim.x - 1, blockDim>>>(d_norm, d_x, (gridDim.x - 1)*blockDim.x);

    // sum of squares for the last incomplete block
    LastBlockSumOfSquare_GPU<<<1, lastBlockSize>>>(d_norm, d_x, N, counter);
	// cudaDeviceSynchronize();
	sqrt_GPU<<<1,1>>>( d_norm );
	// cudaDeviceSynchronize();
}


/// Helper functions for debugging
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
            printf("%f ", x[j+i*num_cols]);

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
void printELLrow_GPU(size_t row, double* value, size_t* index, size_t max_row_size, size_t num_rows, size_t num_cols)
{
		for ( int j = 0 ; j < num_cols ; j++)
			printf("%f ", valueAt(row, j, value, index, max_row_size) );

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
		atomicAdd_double(x, cache[0]);
	}
	__syncthreads();
}


__global__
void LastBlockDotProduct(double* dot, double* x, double* y, size_t starting_index)
{
	int id = threadIdx.x + blockDim.x*blockIdx.x + starting_index;
		
	atomicAdd_double(dot, x[id]*y[id]);
	
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
 

// x += c
__global__
void addVector_GPU(double *x, double *c, size_t num_rows)
{
	int id = blockDim.x * blockIdx.x + threadIdx.x;

	if ( id < num_rows )
		x[id] += c[id];
}


__global__
void transformToELL_GPU(double *array, double *value, size_t *index, size_t max_row_size, size_t num_rows)
{

    int id = blockDim.x * blockIdx.x + threadIdx.x;

    if ( id < num_rows )
    {
        size_t counter = id*max_row_size;
        size_t nnz = 0;
        
			// printf("array = %e\n", array [ 1 ]);
        for ( int j = 0 ; nnz < max_row_size ; j++ )
        {
            if ( array [ j + id*num_rows ] != 0 )
            {
				// printf("array = %e\n", array [ j + id*num_rows ]);
                value [counter] = array [ j + id*num_rows ];
                index [counter] = j;
				// printf("value = %e\n", value[counter]);
                counter++;
                nnz++;
            }
            
            if ( j == num_rows - 1 )
            {
                for ( ; nnz < max_row_size ; counter++ && nnz++ )
                {
                    value [counter] = 0.0;
                    index [counter] = num_rows;
                }
            }
        }
    }
}


std::size_t getMaxRowSize(vector<vector<double>> &array, size_t num_rows, size_t num_cols)
{
	std::size_t max_row_size = 0;

  

	for ( int i = 0; i < num_rows ; i++ )
	{
		std::size_t max_in_row = 0;

		for ( int j = 0 ; j < num_cols ; j++ )
		{
			if ( array[i][j] < -1.0e-8 || array[i][j] > 1.0e-8 )
				max_in_row++;
		}

		if ( max_in_row >= max_row_size )
			max_row_size = max_in_row;
		
	}
	
	return max_row_size;

}

// transforms a 2D array into ELLPACK's vectors value and index
// max_row_size has to be determined prior to this
void transformToELL(vector<vector<double>> &array, vector<double> &value, vector<size_t> &index, size_t max_row_size, size_t num_rows, size_t num_cols )
{
	size_t nnz;
	
	for ( int i = 0 ; i < num_rows ; i++)
    {
		nnz = 0;
			// printf("array = %e\n", array [ 1 ]);
        for ( int j = 0 ; nnz < max_row_size ; j++ )
        {

            if ( array[i][j] < -1.0e-8 || array[i][j] > 1.0e-8 )
            {
				// printf("array = %e\n", array [ j + id*num_rows ]);
				value.push_back(array[i][j]);
				index.push_back(j);
                nnz++;
            }
            
            if ( j == num_cols - 1 )
            {
                for ( ; nnz < max_row_size ; nnz++ )
                {
				value.push_back(0.0);
				index.push_back(num_rows);
                }

            }
        }
	}
	
}

//TEMP:
// sets identity rows and columns of the DOF in which a BC is applied
void applyMatrixBC(vector<vector<double>> &array, size_t index, size_t num_rows, size_t dim)
{
	// index *= dim;

    // for ( int j = 0 ; j < dim ; j++ )
	// {
		for ( int i = 0 ; i < num_rows ; i++ )
		{
			array[i][index] = 0.0;
			array[index][i] = 0.0;
		}

		array[index][index] = 1.0;
	// }
	
}

__host__
void applyMatrixBC(double* value, size_t* index, size_t max_row_size, size_t num_rows, size_t num_cols, size_t dim, size_t bc_index)
{

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

	// nodesPerDim.push_back(N[0]+1);
	// nodesPerDim.push_back(N[1]+1);

	for( int i = 0 ; i < N.size() ; i++ )
		nodesPerDim.push_back(N[i]+1);
	

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

	return bc_index;
}


__host__ 
void applyLoad(vector<double> &b, vector<size_t> N, size_t numLevels, size_t bc_case, size_t dim, double force)
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

__global__
void assembleGrid2D_GPU(
    size_t N,               		// number of elements per row
    size_t dim,             		// dimension
	double* chi,					// the updated design variable value of each element
    double* A_local,      		// local stiffness matrix
    double* value,        // global element's ELLPACK value vector
    size_t* index,        // global element's ELLPACK index vector
    size_t max_row_size,  			// global element's ELLPACK maximum row size
    size_t num_rows,      			// global element's ELLPACK number of rows
    size_t* node_index,      // vector that contains the corresponding global indices of the node's local indices
	size_t p					
)        
{
    int idx = threadIdx.x + blockIdx.x*blockDim.x;
    int idy = threadIdx.y + blockIdx.y*blockDim.y;

	if ( idx < num_rows && idy < num_rows )
    	addAt( 2*node_index[ idx/2 ] + ( idx % 2 ), 2*node_index[idy/2] + ( idy % 2 ), value, index, max_row_size, pow(*chi,p)*A_local[ ( idx + idy * ( 4 * dim ) ) ]  );

    	// addAt( 2*node_index[ idx/2 ] + ( idx % 2 ), 2*node_index[idy/2] + ( idy % 2 ), value, index, max_row_size, A_local[ ( idx + idy * ( 4 * dim ) ) ] );

	// if ( idx == 0 && idy == 0 )
	// 	printf("%e\n", *chi);

}


__global__
void applyMatrixBC_GPU(double* value, size_t* index, size_t max_row_size, size_t bc_index, size_t num_rows)
{
    int idx = threadIdx.x + blockIdx.x*blockDim.x;
    int idy = threadIdx.y + blockIdx.y*blockDim.y;

	if ( idx == bc_index && idy == bc_index )
		setAt( idx, idy, value, index, max_row_size, 1.0 );

}

// CHECK: overkill to use this many threads?
__global__
void applyMatrixBC_GPU_test(double* value, size_t* index, size_t max_row_size, size_t bc_index, size_t num_rows, size_t num_cols)
{
    int idx = threadIdx.x + blockIdx.x*blockDim.x;
    int idy = threadIdx.y + blockIdx.y*blockDim.y;

	// printf("(%d, %d) = %lu, %d, %d\n", idx, idy, bc_index, num_rows, num_cols);
	if ( idx < num_cols && idy < num_rows )
	{
		if ( idx == bc_index && idy == bc_index )
		{
			for ( int i = 0 ; i < num_rows ; i++ )
				setAt( i, idy, value, index, max_row_size, 0.0 );

			for ( int j = 0 ; j < num_cols ; j++ )
				setAt( idx, j, value, index, max_row_size, 0.0 );


			setAt( idx, idy, value, index, max_row_size, 1.0 );
		}
	}
}


// obtain a node's corresponding fine node index
__host__
size_t getFineNode(size_t index, vector<size_t> N, size_t dim)
{
	// check for error
	size_t num_nodes = N[0] + 1;
	for ( int i = 1 ; i < dim ; i++ )
		num_nodes *= (N[i] + 1);
	
	if ( index > num_nodes - 1 )
        throw(runtime_error("Error : Index does not exist on this level"));


	if ( dim == 3 )
	{	
		size_t twoDimSize = (N[0]+1)*(N[1]+1);
		size_t baseindex = index % twoDimSize;
		size_t fine2Dsize = (2*N[0]+1)*(2*N[1]+1);
		size_t multiplier = index/twoDimSize;
		
		return 2*multiplier*fine2Dsize + (2*( baseindex % twoDimSize ) + (ceil)(baseindex/2)*2) ;
		
	}

	else
		return (2 * (ceil)(index / (N[0] + 1)) * (2*N[0] + 1) + 2*( index % (N[0]+1)) );
}


// ////////////////////////////////////////////
// // SMOOTHERS
// ////////////////////////////////////////////

__global__ void Jacobi_Precond_GPU(double* c, double* value, size_t* index, size_t max_row_size, double* r, size_t num_rows, double damp){

	int id = blockDim.x * blockIdx.x + threadIdx.x;

	// B = damp / diag(A);
	if ( id < num_rows )
		c[id] = r[id] * damp / valueAt(id, id, value, index, max_row_size);

}


// ////////////////////////////////////////////
// // SOLVER
// ////////////////////////////////////////////

__global__ 
void checkIterationConditions(bool* foo, size_t* step, double* res, double* res0, double* m_minRes, double* m_minRed, size_t m_maxIter){

	if ( *res > *m_minRes && *res > *m_minRed*(*res0) && (*step) <= m_maxIter )
		*foo = true;

	else
	{
		// printf("false\n");
		// printf("res = %f\n",*res);
		// printf("m_minRes = %f\n",*m_minRes);
		// printf("m_minRed = %f\n",*m_minRed);
		// printf("step = %lu\n",(*step));

		*foo = false;


	}
	
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

		for ( int n = 0; n < num_cols_per_row; n++ )
		{
			int col = index [ num_cols_per_row * id + n ];
			double val = value [ num_cols_per_row * id + n ];
			atomicAdd_double( &r[col], val*x[id] );
		}
	}
}


__global__ 
void printResult_GPU(size_t* step, double* res, double* m_minRes, double* lastRes, double* res0, double* m_minRed)
{
	if(*step < 10)
	printf("    %d    %e    %9.3e    %9.3e    %e    %9.3e    \n", *step, *res, *m_minRes, (*res)/(*lastRes), (*res)/(*res0), *m_minRed);

	else
	printf("   %d    %e    %9.3e    %9.3e    %e    %9.3e    \n", *step, *res, *m_minRes, (*res)/(*lastRes), (*res)/(*res0), *m_minRed);
}

__global__ void addStep(size_t* step){

	++(*step);
}

// BASE SOLVER


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

// A_ = P^T * A * P
__host__
void PTAP(vector<vector<double>> &A_, vector<vector<double>> &A, vector<vector<double>> &P, size_t num_rows, size_t num_rows_)
{
	// temp vectors
	std::vector<std::vector<double>> foo ( num_rows, std::vector <double> (num_rows_, 0.0));


	// foo = A * P
	for ( int i = 0 ; i < num_rows ; i++ )
	{
		for( int j = 0 ; j < num_rows_ ; j++ )
		{
			for ( int k = 0 ; k < num_rows ; k++)
			{
				// cout << "PTAP-ijk = " << i << " " << j << " " << k << endl;
				foo[i][j] += A[i][k] * P[k][j];
			}
		}
	}

	
	

	// A_ = P^T * foo
	for ( int i = 0 ; i < num_rows_ ; i++ )
        {
            for( int j = 0 ; j < num_rows_ ; j++ )
            {
                for ( int k = 0 ; k < num_rows ; k++)
                    A_[i][j] += P[k][i] * foo[k][j];
            }
        }
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	// temp vectors
	// std::vector<std::vector<double>> foo ( num_rows, std::vector <double> (num_rows_, 0.0));

	// double** foo = new double*[num_rows];
	// for(int i = 0; i < num_rows; i++)
	// {
	// 	foo[i] = new double[num_rows_];
	// }
  
	// for ( int i = 0 ; i < num_rows ; i++ )
	// {
	// 	for( int j = 0 ; j < num_rows_ ; j++ )
	// 	{
	// 			foo[i][j] = 0;
	// 	}
	// }


	// // foo = A * P
	// for ( int i = 0 ; i < num_rows ; i++ )
	// {
	// 	for( int j = 0 ; j < num_rows_ ; j++ )
	// 	{
	// 		for ( int k = 0 ; k < num_rows ; k++)
	// 		{
	// 			// cout << "PTAP-ijk = " << i << " " << j << " " << k << endl;
	// 			foo[i][j] += A[i][k] * P[k][j];
	// 		}
	// 	}
	// }

	// // A_ = P^T * foo
	// for ( int i = 0 ; i < num_rows_ ; i++ )
    //     {
    //         for( int j = 0 ; j < num_rows_ ; j++ )
    //         {
    //             for ( int k = 0 ; k < num_rows ; k++)
    //                 A_[i][j] += P[k][i] * foo[k][j];
    //         }
    //     }

	// for ( int i = 0 ; i < num_rows ; i++ )
	// {
	// 	for( int j = 0 ; j < num_rows_ ; j++ )
	// 		cout << foo[i][j] << " ";

	// 	cout << endl;
		
	// }

	// for(int i = 0; i < num_rows; i++)
	// {
	// 	delete [] foo[i];
	// }
	// delete [] foo;









	
	
}


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

	// alpha_temp = () p * z )
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


// df = ( 1/2*omega ) * p * chi^(p-1) * sum(local stiffness matrices)
__global__
void UpdateDrivingForce(double *df, double* uTau, double p, double *chi, double local_volume, size_t N)
{
    int id = blockDim.x * blockIdx.x + threadIdx.x;

    if ( id < N )
        df[id] = uTau[id] * ( local_volume / (2*local_volume) ) * p * pow(chi[id], p - 1);
        // df[id] = uTKu[id] * ( 1 / (2*local_volume) ) * p * pow(chi[id], p - 1);
}

// x[] = u[]^T * A * u[]
__global__
void calcDrivingForce_GPU(double *x, double *u, double* chi, double p, size_t *node_index, double* d_A_local, size_t num_rows, size_t dim)
{
	double temp[24];

	*x = 0;
	for ( int n = 0; n < num_rows; n++ )
	{
		temp[n]=0;
		for ( int m = 0; m < num_rows; m++)
		{
			// converts local node to global node
			int global_col = ( node_index [ m / dim ] * dim ) + ( m % dim ); 


			temp[n] += u[global_col] * d_A_local[ n + m*num_rows ];


		}

	}


	for ( int n = 0; n < num_rows; n++ )
	{
		int global_col = ( node_index [ n / dim ] * dim ) + ( n % dim );
		*x += temp[n] * u[global_col];

		
	}
	
	*x *= 0.5 * p * pow(*chi, p-1);

			// if(n==2)
			// printf("%e = %e x %e\n", *x, temp[n], u[global_col]);
			// 	printf("%e = %e x %e\n", temp[n], u[global_col], d_A_local[ n + m*num_rows ]);
			// if(id==0)
			// printf("%e\n", temp[n]);
			// printf("%e\n", u[global_col]);
			// printf("%d\n", global_col);
		// if(id==0)
		// printf("%e\n", x[id]);

				// if(id==0 && n==1 && m==7)
				// printf("%e = %e x %e\n", u[global_col]*d_A_local[ n + m*num_rows ], u[global_col], d_A_local[ n + m*num_rows ]);

				// if(id==0 && n==1)
				// printf("%e = %e x %e\n", temp[n], u[global_col], d_A_local[ n + m*num_rows ]);
							// printf("%e\n", u[global_col]);
			// printf("%e\n", x[id]);
			// printf("%d\n", global_col);
    
}


// calculate the driving force per element
__host__
void calcDrivingForce(
    double *df,             // driving force
    double *chi,            // design variable
    double p,               // penalization parameter
    double *uTAu,           // dummy/temp vector
    double *u,              // elemental displacement vector
    vector<size_t*> node_index,
	double* d_A_local,
    size_t num_rows,        // local ELLPack stiffness matrix's number of rows
    dim3 gridDim,           // grid and 
    dim3 blockDim,
	const size_t dim,
	size_t numElements)          // block sizes needed for running CUDA kernels
{
	
	// printVector_GPU<<<1,20>>>( u, 20);

	// calculate the driving force in each element ( 1 element per thread )
    // df[] = 0.5 * p * pow(chi.p-1) - u[]^T * A * u[]
	
	for ( int i = 0 ; i < numElements; i++ )
	    calcDrivingForce_GPU<<<1, 1>>>(&df[i], u, &chi[i], p, node_index[i], d_A_local, num_rows, dim);

    cudaDeviceSynchronize();
	// printVector_GPU<<<1,numElements>>>( df, numElements);




	// // calculate the driving force in each element
    // sumOfVector_GPU<<<gridDim, blockDim>>>(df, uTAu, num_rows);
    // cudaDeviceSynchronize();
    
	//TODO: det_J not implemented yet
	// df[] *= ( 1 / 2*omega ) * ( p * pow(chi[], p - 1 ) * det(J)
    // UpdateDrivingForce<<<gridDim,blockDim>>>(df, uTAu, p, chi, local_volume, num_rows);
	
}

__global__ 
void sumOfVector_GPU(double* sum, double* x, size_t n)
{
    int id = blockDim.x * blockIdx.x + threadIdx.x;
	int stride = blockDim.x*gridDim.x;
    
    // if ( id < n )
    // printf("%d : %e\n", id, x[id]);

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
		atomicAdd_double(sum, cache[0]);
}


__host__ 
void TestcalcDrivingForce(
	double *df, 
	double *chi, 
	double p,
	double *u, 
	size_t* node_index, 
	double* d_A_local, 
	size_t num_rows, 
	dim3 gridDim, 
	dim3 blockDim,
	size_t numElements)
{



}

__global__ 
void calcDrivingForce_(
    double *df,             // driving force
    double *chi,            // design variable
    double p,               // penalization parameter
    double *u,              // elemental displacement vector
    size_t* node_index,
	double* d_A_local,
    size_t num_rows,        // local ELLPack stiffness matrix's number of rows
    size_t dim)          
{
	int id = blockDim.x * blockIdx.x + threadIdx.x;

	__shared__ double uTAu_[1024];

    if ( id < num_rows )
    {
        uTAu_[id] = 0;

		// uTAu = uT * A
        for ( int n = 0; n < num_rows; n++ )
		{
			// converts local node to global node
            int global_col = ( node_index [ n / dim ] * dim ) + ( n % dim ); 
            uTAu_[id] += u[global_col] * d_A_local[ id + n*num_rows ];

        }

		// uTAu *= u
        uTAu_[id] *= u[ ( node_index [ id / dim ] * dim ) + ( id % dim ) ];


		df[id] = uTAu_[id] * (p) * pow(chi[id], (p-1));
    }
	
}

__device__
double laplacian_GPU( double *array, size_t ind, size_t N )
{
    double value = 4.0 * array[ind];

	// if ( ind == 0 )
	// {
	// 	printf("%f\n", *array);
	// 	printf("%lu\n", N);
	// }
	
    // east element
    if ( (ind + 1) % N != 0 )
        value += -1.0 * array[ind + 1];
	else
		value += -1.0 * array[ind];
    
    // north element
    if ( ind + N < N*N )    // TODO: N*N --> dim
        value += -1.0 * array[ind + N];
	else
		value += -1.0 * array[ind];
	

    // west element
    if ( ind % N != 0 )
        value += -1.0 * array[ind - 1];
	else
		value += -1.0 * array[ind];

    // south element
    if ( ind >= N )
        value += -1.0 * array[ind - N];
	else
		value += -1.0 * array[ind];

    return value;
}

__global__ 
void calcLambdaUpper(double *df_array, double *max, int *mutex, double* beta, double *chi, double* eta, unsigned int N, unsigned int numElements)
{
	unsigned int index = threadIdx.x + blockIdx.x*blockDim.x;
	unsigned int stride = gridDim.x*blockDim.x;
	unsigned int offset = 0;

	__shared__ double cache[1024];

    *max = -1.0e9;
    double temp = -1.0e9;
    
	while(index + offset < numElements){
        
		//TODO:DEBUG:
        // temp = fmaxf(temp, ( df_array[index + offset] + ( *beta * laplacian_GPU( chi, index, N ) ) + *eta ) );
        temp = fmaxf(temp, ( df_array[index + offset] + *eta ) );
         
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
void calcLambdaLower(double *df_array, double *min, int *mutex, double* beta, double *chi, double* eta, unsigned int N, unsigned int numElements)
{
	unsigned int index = threadIdx.x + blockIdx.x*blockDim.x;
	unsigned int stride = gridDim.x*blockDim.x;
	unsigned int offset = 0;

	__shared__ double cache[1024];

    *min = 1.0e9;
    double temp = 1.0e9;
    

	while(index + offset < numElements){
        //TODO:DEBUG:
		// temp = fminf(temp, ( df_array[index + offset] + ( *beta * laplacian_GPU( chi, index, N ) ) - *eta ) );
        temp = fminf(temp, ( df_array[index + offset] - *eta ) );
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

__global__
void calcChiTrial(   
    double *chi, 
    double *df, 
    double *lambda_trial, 
    double del_t,
    double* eta,
    double* beta,
    double* chi_trial,
    size_t N,
    size_t numElements
)
{
    unsigned int id = threadIdx.x + blockIdx.x*blockDim.x;
    
    __shared__ double del_chi[1024];


    if ( id < numElements )
    {
		// printf("%d : %e \n", id, del_chi[id]);
		// printf("%e \n", *eta);

		//TODO:DEBUG:
        // del_chi[id] = ( del_t / *eta ) * ( df[id] - *lambda_trial + (*beta)*( laplacian_GPU( chi, id, N ) ) );
        del_chi[id] = ( del_t / *eta ) * ( df[id] - *lambda_trial );
        

        if ( del_chi[id] + chi[id] > 1 )
        chi_trial[id] = 1;
        
        else if ( del_chi[id] + chi[id] < 1e-9 )
        chi_trial[id] = 1e-9;
        
        else
        chi_trial[id] = del_chi[id] + chi[id];
        
        // printf("%d %f \n", id, chi_trial[id]);
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



// NOTE: shelved for now
__global__ 
void int_g_p(double* d_temp, double* d_df, double local_volume, size_t numElements)
{
	// unsigned int id = threadIdx.x + blockIdx.x*blockDim.x;

	// if( id < numElements)
	// {
	// 	// calculate g of element
	// 	d_temp[id] = (d_chi[id] - 1e-9)*(1-d_chi[id]) * d_df[id] * local_volume; 

	// }

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

		// atomicAdd_double(&d_temp[0], int_g_p[id]);
		// atomicAdd_double(&d_temp[1], int_g[id]);

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
void calc_g_GPU(double*g, double* chi, size_t numElements)
{
	unsigned int id = threadIdx.x + blockIdx.x*blockDim.x;

	if (id < numElements)
	{
		g[id] = (chi[id] - 1e-9)*(1-chi[id]);
	}
}


__global__ 
void calcSum_df_g_GPU(double* sum, double* df, double* g, size_t n)
{
    int id = blockDim.x * blockIdx.x + threadIdx.x;
	int stride = blockDim.x*gridDim.x;
    
    // if ( id < n )
    // printf("%d : %e\n", id, x[id]);

	__shared__ double cache[1024];
    cache[threadIdx.x] = 0;
    
	double temp = 0.0;
	while(id < n)
	{
		temp += df[id]*g[id];
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
		atomicAdd_double(sum, cache[0]);
}


__host__
void calcP_w(double* p_w, double* df, double* chi, double* g, double* df_g, size_t numElements)
{	
	dim3 gridDim;
	dim3 blockDim;

	calculateDimensions(numElements, gridDim, blockDim);
	calc_g_GPU<<<gridDim, blockDim>>>(g, chi, numElements);


	// calculate p_w = sum(g)
	// using p_w as a temp
	sumOfVector_GPU<<<gridDim, blockDim>>>(df_g, g, numElements);
	calcSum_df_g_GPU<<<gridDim, blockDim>>>(p_w, df, g, numElements);
	
	divide_GPU<<<1,1>>>(p_w, p_w, df_g);
	// cudaDeviceSynchronize();
	// printVector_GPU<<<gridDim,blockDim>>>( df, numElements );



}


__global__ void calcEtaBeta( double* eta, double* beta, double etastar, double betastar, double* p_w )
{
	unsigned int id = threadIdx.x + blockIdx.x*blockDim.x;

	if ( id == 0 )	
		*eta = etastar * (*p_w);

	if ( id == 1 )
		*beta = betastar * (*p_w);

}

__global__ void RA(	
	double* r_value, 		// restriction matrix's
	size_t* r_index, 		// ELLPACK vectors
	size_t r_max_row_size, 	
	double* value, 			// global stiffness matrix's
	size_t* index, 			// ELLPACK vectors
	size_t max_row_size,	
	double* temp_matrix, 	// empty temp matrix
	size_t num_rows, 		// no. of rows of temp matrix
	size_t num_cols			// no. of cols of temp matrix
)
{
	unsigned int idx = threadIdx.x + blockIdx.x*blockDim.x;
	unsigned int idy = threadIdx.y + blockIdx.y*blockDim.y;

	if ( idx < num_cols && idy < num_rows )
	{	
		
		for ( int j = 0 ; j < num_cols ; ++j )
			temp_matrix[idx + idy*num_cols] += valueAt(idy, j, r_value, r_index, r_max_row_size) * valueAt(j, idx, value, index, max_row_size);
		
	}

}


__global__ void AP(	
	double* value, 			// global stiffness matrix's
	size_t* index,			// ELLPACK vectors
	size_t max_row_size, 
	double* p_value, 		// prolongation matrix's
	size_t* p_index, 		// ELLPACK vectors
	size_t p_max_row_size, 
	double* temp_matrix, 	// temp_matrix = R*A
	size_t num_rows, 		// no. of rows of temp matrix
	size_t num_cols			// no. of cols of temp matrix
)
{
	unsigned int idx = threadIdx.x + blockIdx.x*blockDim.x;
	unsigned int idy = threadIdx.y + blockIdx.y*blockDim.y;

	if ( idx < num_cols && idy < num_rows )
	{	
		for ( int j = 0 ; j < num_cols ; ++j )
			addAt( idx, idy, value, index, max_row_size, temp_matrix[j + idy*num_cols] * valueAt(j, idx, p_value, p_index, p_max_row_size) );

	}

}


// A_coarse = R * A_fine * P
// TODO: not optimized yet
__host__ void RAP(	vector<double*> value, vector<size_t*> index, vector<size_t> max_row_size, 
					vector<double*> r_value, vector<size_t*> r_index, vector<size_t> r_max_row_size, 
					vector<double*> p_value, vector<size_t*> p_index, vector<size_t> p_max_row_size, 
					double* temp_matrix,
					vector<size_t> num_rows, 
					size_t lev)
{
	

	// dim3 gridDim(2,2,1);
    // dim3 blockDim(32,32,1);
	dim3 gridDim;
    dim3 blockDim;
	calculateDimensions2D( num_rows[lev], num_rows[lev], gridDim, blockDim);
	

	// temp_matrix = R * A_fine
	RA<<<gridDim,blockDim>>>(r_value[lev-1], r_index[lev-1], r_max_row_size[lev-1], value[lev], index[lev], max_row_size[lev], temp_matrix, num_rows[lev-1], num_rows[lev]);
	cudaDeviceSynchronize();

	// calculateDimensions2D( num_rows[0] * num_rows[0], gridDim, blockDim);
	AP<<<gridDim,blockDim>>>( value[lev-1], index[lev-1], max_row_size[lev-1], p_value[lev-1], p_index[lev-1], p_max_row_size[lev-1], temp_matrix, num_rows[lev-1], num_rows[lev]);
	cudaDeviceSynchronize();

	// cout << blockDim.x << endl;
	// cout << blockDim.y << endl;
	// cout << blockDim.z << endl;

	// A_coarse = temp_matrix * P

	// printVector_GPU<<<1, 18*50>>>( temp_matrix, 18*50);

	// printELL_GPU<<<1, 1>>> (value[2], index[2], max_row_size[2], num_rows[2], num_rows[2]);
	// printELL_GPU<<<1, 1>>> (r_value[0], r_index[0], r_max_row_size[0], 8, 18);



	// for ( int i = 0 ; i < num_rows ; i++ )
	// {
	// 	for( int j = 0 ; j < num_rows_ ; j++ )
	// 	{
	// 		for ( int k = 0 ; k < num_rows ; k++)
	// 			foo[i][j] += A[i][k] * P[k][j];
	// 	}
	// }

	// // A_ = P^T * foo
	// for ( int i = 0 ; i < num_rows_ ; i++ )
    //     {
    //         for( int j = 0 ; j < num_rows_ ; j++ )
    //         {
    //             for ( int k = 0 ; k < num_rows ; k++)
    //                 A_[i][j] += P[k][i] * foo[k][j];
    //         }
    //     }

}

__global__ void checkTDOConvergence(bool* foo, double rho, double* rho_trial)
{
	if ( abs(rho - *rho_trial) < 1e-7 )
		*foo = false;
}


__global__
void bar(size_t x, size_t y, double* vValue, size_t* vIndex, size_t max_row_size)
{
    for(size_t k = 0; k < max_row_size; ++k)
    {
        if(vIndex[x * max_row_size + k] == y)
		{
            printf("%e\n", vValue[x * max_row_size + k]);

		}
    }

    printf("tak jumpa, so zero\n");
}

__global__ void assembleGlobal_GPU(size_t* index, size_t Nx, size_t Ny, size_t max_row_size, size_t num_rows)
{
	unsigned int id = threadIdx.x + blockIdx.x*blockDim.x;

	int counter = 0;
	int dim = 2;

	int base_id = (id - id%dim);
	// NOTE: base_id = (id - id%dim)

	// DEBUG: testing row 1
	
	// south-west
	if ( id  >= (Nx + 1)*dim && (id) % ((Nx + 1)*dim) >= dim )
	{
		for(int i = 0 ; i < dim ; i++)
		{
			index[counter + id*max_row_size] = (id - id%dim) - (Nx+1)*dim - dim + i;
			counter++;
		}
	}

	// south
	if ( id  >= (Nx + 1)*dim )
	{
		for(int i = 0 ; i < dim ; i++)
		{
			index[counter + id*max_row_size] = (id - id%dim) - (Nx+1)*dim + i;
			counter++;
		}
	}

	// south-east
	if ( id  >= (Nx + 1)*dim && (base_id) % ((Nx*dim) + (base_id/(2*(Nx+1)))*dim*(Nx+1)) != 0 )
	{
		for(int i = 0 ; i < dim ; i++)
		{
			index[counter + id*max_row_size] = (id - id%dim) - (Nx+1)*dim + dim + i;
			counter++;
		}
	}

	// west
	if ( (id) % ((Nx + 1)*dim) >= dim )
	{

		for(int i = 0 ; i < dim ; i++)
		{
			index[counter + id*max_row_size] = (id - id%dim) - dim + i;
			counter++;
		}
	}

	// origin
	for(int i = 0 ; i < dim ; i++)
		{
			index[counter + id*max_row_size] = (id - id%dim) + i;
			counter++;
		}

	// if ( (id) % (Nx*dim) != Nx*dim )

	// if( id == 41 )
	// 	printf("%d\n", (base_id) % ((Nx*dim) + (base_id/(2*(Nx+1)))*dim*(Nx+1)) );
		// printf("%d\n", (base_id/(2*(Nx+1)))*dim*(Nx+1) );

	// east
	if ( base_id == 0 || (base_id) % ((Nx*dim) + (base_id/(2*(Nx+1)))*dim*(Nx+1)) != 0 )
	{
		for(int i = 0 ; i < dim ; i++)
		{
			index[counter + id*max_row_size] = (id - id%dim) + dim + i;
			counter++;
		}
	}

	// north-west
	if ( id < (Nx+1)*(Ny)*dim && (id) % ((Nx + 1)*dim) >= dim )
	{
		for(int i = 0 ; i < dim ; i++)
		{
			index[counter + id*max_row_size] = (id - id%dim) + (Nx+1)*dim - dim + i;
			counter++;
		}
	}

	// north
	if ( id < (Nx+1)*(Ny)*dim )
	{
		for(int i = 0 ; i < dim ; i++)
		{
			index[counter + id*max_row_size] = (id - id%dim) + (Nx+1)*dim + i;
			counter++;
		}
	}

	// north-east
	if ( base_id == 0 || id < (Nx+1)*(Ny)*dim && (base_id) % ((Nx*dim) + (base_id/(2*(Nx+1)))*dim*(Nx+1)) != 0 )
	{
		for(int i = 0 ; i < dim ; i++)
		{
			index[counter + id*max_row_size] = (id - id%dim) + (Nx+1)*dim + dim + i;
			counter++;
		}
	}

	// if ( id == 0 )
	// {
	// 	// printf("%d\n", 1 - 1%2 );
	// 	// printf("%d\n", 3 - 3%2 );
	// 	// printf("%d\n", 2%2 );
	// 	for ( int i = 0 ; i < max_row_size ; ++i )
	// 	printf("%lu\n", index[id*max_row_size+i]);

	// }
}