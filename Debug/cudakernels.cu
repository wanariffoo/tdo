
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
        if(vIndex[y * max_row_size + k] == x )
        {
            vValue[y * max_row_size + k] = value;
                k = max_row_size; // to exit for loop
		}
    }
}

__device__
void setAt_( size_t x, size_t y, double* vValue, size_t* vIndex, size_t num_cols, size_t max_row_size, double value )
{
    for(size_t k = 0; k < max_row_size; ++k)
    {
        if(vIndex[y * max_row_size + k] == x && k < num_cols)
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
	{
		size_t local_num_cols = pow(2,dim) * dim;
    	addAt( dim*node_index[ idx/dim ] + ( idx % dim ), dim*node_index[idy/dim] + ( idy % dim ), value, index, max_row_size, pow(*chi,p)*A_local[ ( idx + idy*local_num_cols ) ]  );
	}
    	// addAt( 2*node_index[ idx/2 ] + ( idx % 2 ), 2*node_index[idy/2] + ( idy % 2 ), value, index, max_row_size, pow(*chi,p)*A_local[ ( idx + idy * ( 4 * dim ) ) ]  );

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

__global__
void applyProlMatrixBC_GPU(double* value, size_t* index, size_t max_row_size, size_t bc_index, size_t num_rows, size_t num_cols)
{
	for ( int i = 0 ; i < num_rows ; i++ )
	{
		if ( valueAt(i, bc_index, value, index, max_row_size) != 1.0 )
			setAt( bc_index, i, value, index, max_row_size, 0.0 );
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
		size_t base_idx = baseindex % (N[0]+1);
		size_t fine2Dsize = (2*N[0]+1)*(2*N[1]+1);
		size_t multiplier = index/twoDimSize;
		
		// return 2*multiplier*fine2Dsize + (2*( baseindex % twoDimSize ) + (ceil)(baseindex/2)*2) ;
		return 2*base_idx + (baseindex/(N[0]+1))*2*(2*N[0] + 1) + 2*fine2Dsize*multiplier;
		
		
	}

	else
		return (2 * (ceil)(index / (N[0] + 1)) * (2*N[0] + 1) + 2*( index % (N[0]+1)) );
}

// input the coarse node's "index" to obtain the node's corresponding fine node index
__device__
size_t getFineNode_GPU(size_t index, size_t Nx, size_t Ny, size_t Nz, size_t dim)
{
	
	// size_t num_nodes = (Nx + 1)*(Ny + 1)*(Nz + 1);
	
	if ( dim == 3 )
	{	
		size_t twoDimSize = (Nx+1)*(Ny+1);
		size_t baseindex = index % twoDimSize;
		size_t base_idx = baseindex % (Nx+1);
		size_t fine2Dsize = (2*Nx+1)*(2*Ny+1);
		size_t multiplier = index/twoDimSize;
		
		return 2*base_idx + (baseindex/(Nx+1))*2*(2*Nx + 1) + 2*fine2Dsize*multiplier;
		// return 2*multiplier*fine2Dsize + (2*( baseindex  ) + (baseindex/2)*2) ;
		// return 2*multiplier*fine2Dsize + (2*( baseindex % twoDimSize ) + (baseindex/2)*2) ;
		
	}

	else
		return (2 * (index / (Nx + 1)) * (2*Nx + 1) + 2*( index % (Nx+1)) );
}


__global__ void fillRestMatrix(double* r_value, size_t* r_index, size_t r_max_row_size, double* p_value, size_t* p_index, size_t p_max_row_size, size_t num_rows, size_t num_cols)
{
	int idx = threadIdx.x + blockIdx.x*blockDim.x;
    int idy = threadIdx.y + blockIdx.y*blockDim.y;

	if ( idx < num_cols && idy < num_rows )
		setAt_( r_index[idx + idy*r_max_row_size], idy, r_value, r_index, num_cols, r_max_row_size, valueAt(r_index[idx + idy*r_max_row_size], idy, p_value, p_index, p_max_row_size));

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
void checkIterationConditions(bool* foo, size_t* step, double* res, double* res0, double* m_minRes, double* m_minRed, size_t m_maxIter)
{
	if ( *res > *m_minRes && *res > *m_minRed*(*res0) && (*step) <= m_maxIter )
	{
		// printf("%lu : %e (%e), %e (%e)\n", (*step), *res, *m_minRes, (*res)/(*res0), *m_minRed);
		*foo = true;

	}

	else
		*foo = false;
}


__global__ 
void checkIterationConditionsBS(bool* foo, size_t* step, size_t m_maxIter, double* res, double* m_minRes)
{
	// if ( *res > *m_minRes && (*step) <= m_maxIter )
	if ( *res > 1e-10 && (*step) <= m_maxIter )
	{
		// printf("%lu : %e (%e), %e (%e)\n", (*step), *res, *m_minRes, (*res)/(*res0), *m_minRed);
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
void calcDrivingForce_GPU(double *x, double *u, double* chi, double p, size_t *node_index, double* d_A_local, size_t num_rows, size_t dim, double local_volume)
{
	double temp[24]; //CHECK:

	*x = 0;
	for ( int n = 0; n < num_rows; n++ )
	{
		temp[n]=0;
		for ( int m = 0; m < num_rows; m++)
		{
			// converts local node to global node
			int global_col = ( node_index [ m / dim ] * dim ) + ( m % dim ); 
			// printf("u[%d] = %f\n", global_col, u[global_col]);


			temp[n] += u[global_col] * d_A_local[ n + m*num_rows ];


		}

	}


	for ( int n = 0; n < num_rows; n++ )
	{
		int global_col = ( node_index [ n / dim ] * dim ) + ( n % dim );
		*x += temp[n] * u[global_col];

		
	}
	
	*x *= 0.5 * p * pow(*chi, p-1) / local_volume;
    
}


// calculate the driving force per element
__host__
void calcDrivingForce(
    double *df,             	// driving force
    double *chi,            	// design variable
    double p,               	// penalization parameter
    double *uTAu,           	// dummy/temp vector
    double *u,              	// elemental displacement vector
    vector<size_t*> node_index,
	double* d_A_local,
    size_t num_rows,        	// local ELLPack stiffness matrix's number of rows
    dim3 gridDim,           	// grid and 
    dim3 blockDim,
	const size_t dim,
	size_t numElements,			// block sizes needed for running CUDA kernels
	double local_volume
	)
{
	// calculate the driving force in each element ( 1 element per thread )
    // df[] = (0.5/local_volume) * p * pow(chi,p-1) - u[]^T * A_local * u[]
	for ( int i = 0 ; i < numElements; i++ )
	    calcDrivingForce_GPU<<<1, 1>>>(&df[i], u, &chi[i], p, node_index[i], d_A_local, num_rows, dim, local_volume);

    cudaDeviceSynchronize();

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

	double uTAu;

    if ( id < num_rows )
    {
        uTAu = 0;

		// uTAu = uT * A
        for ( int n = 0; n < num_rows; n++ )
		{
			// converts local node to global node
            int global_col = ( node_index [ n / dim ] * dim ) + ( n % dim ); 
            uTAu += u[global_col] * d_A_local[ id + n*num_rows ];

        }

		// uTAu *= u
        uTAu *= u[ ( node_index [ id / dim ] * dim ) + ( id % dim ) ];


		df[id] = uTAu * (p) * pow(chi[id], (p-1));
    }
	
}

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
        
		//TODO:DEBUG:
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
		atomicAdd_double(sum, cache[0]);
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
											//TODO: R matrix, no need valueAt, direct lookup
	}

}


__global__ void AP(	
	double* value, 			// coarse global stiffness matrix's
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
	
}

// TODO: CHECK:
// returns value at (row, col) of matrix multiplication A*B
__device__ double matMul(size_t row, size_t col, 
						 double* A_value, size_t* A_index, size_t A_max_row_size, size_t A_num_rows,
						 double* B_value, size_t* B_index, size_t B_max_row_size, size_t b_num_rows	)
{
	__shared__ double value;

	value = 0;

	for(int i = 0 ; i < A_max_row_size ; i++ )
	{
		value += valueAt(row, A_index[i+A_max_row_size*row], A_value, A_index, A_max_row_size) * valueAt(A_index[i+A_max_row_size*row], col, B_value, B_index, B_max_row_size);

		// printf("%f %f\n ", valueAt(row, A_index[i], A_value, A_index, A_max_row_size), valueAt(A_index[i], col, B_value, B_index, B_max_row_size)  );
			// printf("%f\n ", valueAt(B_index[i], col, B_value, B_index, B_max_row_size) );
	}

	// printf("%f\n ", value );
	return value;


}


// A_coarse = R * A_fine * P
__global__ void RAP_(	double* value, size_t* index, size_t max_row_size, size_t num_rows,
						double* value_, size_t* index_, size_t max_row_size_, size_t num_rows_, 
						double* r_value, size_t* r_index, size_t r_max_row_size, 
						double* p_value, size_t* p_index, size_t p_max_row_size, 
						size_t lev)
{
		double RAP = 0;

		unsigned int col = threadIdx.x + blockIdx.x*blockDim.x;
		unsigned int row = threadIdx.y + blockIdx.y*blockDim.y;
		

		if ( row < num_rows_ && col < num_rows_ )
		{
			for ( int i = 0 ; i < r_max_row_size ; i++ )
				RAP += matMul(row, r_index[i + col*r_max_row_size], r_value, r_index, r_max_row_size, num_rows_, value, index, max_row_size, num_rows ) * valueAt(r_index[i+col*r_max_row_size], col, p_value, p_index, p_max_row_size);
			setAt( col, row, value_, index_, max_row_size_, RAP );
		}

}

__global__ void checkTDOConvergence(bool* foo, double rho, double* rho_trial)
{
	if ( abs(rho - *rho_trial) < 1e-7 )
		*foo = false;
}



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

	}
}

__global__ void fillIndexVector3D_GPU(size_t* index, size_t Nx, size_t Ny, size_t Nz, size_t max_row_size, size_t num_rows)
{
	unsigned int id = threadIdx.x + blockIdx.x*blockDim.x;

	int counter = 0;
	int dim = 3;

	if ( id < num_rows )
	{
	
	size_t base_id = (id - id%dim);
	size_t baseid_2D = (id) % ((Nx + 1)*(Ny + 1)*dim);
	size_t gridsize_2D = (Nx+1)*(Ny+1)*dim;

	bool prev_layer = (id >= (Nx+1)*(Ny+1)*dim);
	bool next_layer = (id < (Nx+1)*(Ny+1)*(Nz)*dim);
	bool south = ((id) % ((Nx + 1)*(Ny + 1)*dim)  >= (Nx + 1)*dim);
	bool north = ((id) % ((Nx + 1)*(Ny + 1)*dim) < (Nx+1)*(Ny)*dim);
	bool west = ((id) % ((Nx + 1)*dim) >= dim);
	bool east = ((base_id) % ((Nx*dim) + (base_id/(3*(Nx+1)))*dim*(Nx+1)) != 0);
	





	//// previous layer

		
		// south-west
		// if ( id >= (Nx+1)*(Ny+1)*dim && (id) % ((Nx + 1)*(Ny + 1)*dim)  >= (Nx + 1)*dim && (id) % ((Nx + 1)*dim) >= dim )
		if ( prev_layer && south && west )
		{
			for(int i = 0 ; i < dim ; i++)
			{
				index[counter + id*max_row_size] = (id - id%dim) - (Nx+1)*dim - dim + i - gridsize_2D;
				counter++;
			}
		}

		// south
		// if ( id >= (Nx+1)*(Ny+1)*dim && (id) % ((Nx + 1)*(Ny + 1)*dim)  >= (Nx + 1)*dim )
		if ( prev_layer && south )
		{
			for(int i = 0 ; i < dim ; i++)
			{
				index[counter + id*max_row_size] = (id - id%dim) - (Nx+1)*dim + i - gridsize_2D;
				counter++;
			}

		}

		// south-east
		// if ( id >= (Nx+1)*(Ny+1)*dim && (id) % ((Nx + 1)*(Ny + 1)*dim)  >= (Nx + 1)*dim && (base_id) % ((Nx*dim) + (base_id/(3*(Nx+1)))*dim*(Nx+1)) != 0 )
		if ( prev_layer && south && east )
		{	
			for(int i = 0 ; i < dim ; i++)
			{
				index[counter + id*max_row_size] = (id - id%dim) - (Nx+1)*dim + dim + i - gridsize_2D;
				counter++;
			}

		}

		// west
		// if ( id >= (Nx+1)*(Ny+1)*dim && (id) % ((Nx + 1)*dim) >= dim )
		if ( prev_layer && west )
		{

			for(int i = 0 ; i < dim ; i++)
			{
				index[counter + id*max_row_size] = (id - id%dim) - dim + i - gridsize_2D;
				counter++;
			}

		}

		// origin
		// if ( id >= (Nx+1)*(Ny+1)*dim )
		if ( prev_layer )
		{

			for(int i = 0 ; i < dim ; i++)
			{
				index[counter + id*max_row_size] = (id - id%dim) + i - gridsize_2D;
				counter++;
			}
		}



		// east
		// if ( (base_id == 0 && (base_id) % ((Nx*dim) + (base_id/(3*(Nx+1)))*dim*(Nx+1)) != 0 ) || id >= (Nx+1)*(Ny+1)*dim && (base_id) % ((Nx*dim) + (base_id/(3*(Nx+1)))*dim*(Nx+1)) != 0)
		if ( prev_layer && east )
		{

			for(int i = 0 ; i < dim ; i++)
				{
					index[counter + id*max_row_size] = (id - id%dim) + dim + i - gridsize_2D;
					counter++;
				}
		}




		// north-west
		// if ( id >= (Nx+1)*(Ny+1)*dim && (id) % ((Nx + 1)*(Ny + 1)*dim) < (Nx+1)*(Ny)*dim && (id) % ((Nx + 1)*dim) >= dim )
		if ( prev_layer && north && west )
		{
			for(int i = 0 ; i < dim ; i++)
			{
				index[counter + id*max_row_size] = (id - id%dim) + (Nx+1)*dim - dim + i - gridsize_2D;
				counter++;
			}

		}

		// north
		// if ( id >= (Nx+1)*(Ny+1)*dim && (id) % ((Nx + 1)*(Ny + 1)*dim) < (Nx+1)*(Ny)*dim )
		if ( prev_layer && north )
		{
			for(int i = 0 ; i < dim ; i++)
			{
				index[counter + id*max_row_size] = (id - id%dim) + (Nx+1)*dim + i - gridsize_2D;
				counter++;
			}

		}

		// north-east
		// if ( ((id) % ((Nx + 1)*(Ny + 1)*dim) && base_id == 0 && (base_id) % ((Nx*dim) + (base_id/(3*(Nx+1)))*dim*(Nx+1)) != 0 ) || ( (id) % ((Nx + 1)*(Ny + 1)*dim) && id >= (Nx+1)*(Ny+1)*dim && (base_id) % ((Nx*dim) + (base_id/(3*(Nx+1)))*dim*(Nx+1)) != 0) )
		if ( prev_layer && north && east )
		{
			for(int i = 0 ; i < dim ; i++)
			{
				index[counter + id*max_row_size] = (id - id%dim) + (Nx+1)*dim + dim + i - gridsize_2D;
				counter++;
			}

		}










	//// current layer
		
		// south-west
		// if ( (id) % ((Nx + 1)*(Ny + 1)*dim)  >= (Nx + 1)*dim && (id) % ((Nx + 1)*dim) >= dim )
		if ( south && west )
		{

			for(int i = 0 ; i < dim ; i++)
			{
				index[counter + id*max_row_size] = (id - id%dim) - (Nx+1)*dim - dim + i;
				counter++;
			}
		}

		// south
		// if ( (id) % ((Nx + 1)*(Ny + 1)*dim)  >= (Nx + 1)*dim )
		if ( south )
		{
			for(int i = 0 ; i < dim ; i++)
			{
				index[counter + id*max_row_size] = (id - id%dim) - (Nx+1)*dim + i;
				counter++;
			}

		}

		// south-east
		// if ( (id) % ((Nx + 1)*(Ny + 1)*dim)  >= (Nx + 1)*dim && (base_id) % ((Nx*dim) + (base_id/(3*(Nx+1)))*dim*(Nx+1)) != 0 )
		if ( south && east )
		{	
			for(int i = 0 ; i < dim ; i++)
			{
				index[counter + id*max_row_size] = (id - id%dim) - (Nx+1)*dim + dim + i;
				counter++;
			}

		}

		// west
		// if ( (id) % ((Nx + 1)*dim) >= dim )
		if ( west )
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



		// east
		// if ( base_id == 0 || (base_id) % ((Nx*dim) + (base_id/(3*(Nx+1)))*dim*(Nx+1)) != 0 )
		if ( base_id == 0 || east )
		{

			for(int i = 0 ; i < dim ; i++)
				{
					index[counter + id*max_row_size] = (id - id%dim) + dim + i;
					counter++;
				}
		}




		// north-west
		// if ( (id) % ((Nx + 1)*(Ny + 1)*dim) < (Nx+1)*(Ny)*dim && (id) % ((Nx + 1)*dim) >= dim )
		if ( north && west )
		{
			for(int i = 0 ; i < dim ; i++)
			{
				index[counter + id*max_row_size] = (id - id%dim) + (Nx+1)*dim - dim + i;
				counter++;
			}

		}

		// north
		// if ( (id) % ((Nx + 1)*(Ny + 1)*dim) < (Nx+1)*(Ny)*dim )
		if ( north )
		{
			for(int i = 0 ; i < dim ; i++)
			{
				index[counter + id*max_row_size] = (id - id%dim) + (Nx+1)*dim + i;
				counter++;
			}

		}

		// north-east
		// if ( (base_id == 0 ) || ( (id) % ((Nx + 1)*(Ny + 1)*dim) < (Nx+1)*(Ny)*dim && (base_id) % ((Nx*dim) + (base_id/(3*(Nx+1)))*dim*(Nx+1)) != 0 ) )
		if ( base_id == 0 || (north && east ) )
		{
			for(int i = 0 ; i < dim ; i++)
			{
				index[counter + id*max_row_size] = (id - id%dim) + (Nx+1)*dim + dim + i;
				counter++;
			}

		}


	//// next layer
	
		// south-west
		// if ( id < (Nx+1)*(Ny+1)*(Nz)*dim && (id) % ((Nx + 1)*(Ny + 1)*dim)  >= (Nx + 1)*dim && (id) % ((Nx + 1)*dim) >= dim )
		if ( next_layer && south && west )
		{
			for(int i = 0 ; i < dim ; i++)
			{
				index[counter + id*max_row_size] = (id - id%dim) - (Nx+1)*dim - dim + i + gridsize_2D;
				counter++;
			}
		}

		// south
		// if ( id < (Nx+1)*(Ny+1)*(Nz)*dim && (id) % ((Nx + 1)*(Ny + 1)*dim)  >= (Nx + 1)*dim )
		if ( next_layer && south )
		{
			for(int i = 0 ; i < dim ; i++)
			{
				index[counter + id*max_row_size] = (id - id%dim) - (Nx+1)*dim + i + gridsize_2D;
				counter++;
			}

		}

		// south-east
		// if ( id < (Nx+1)*(Ny+1)*(Nz)*dim && (id) % ((Nx + 1)*(Ny + 1)*dim)  >= (Nx + 1)*dim && (base_id) % ((Nx*dim) + (base_id/(3*(Nx+1)))*dim*(Nx+1)) != 0 )
		if ( next_layer && south && east )
		{	
			for(int i = 0 ; i < dim ; i++)
			{
				index[counter + id*max_row_size] = (id - id%dim) - (Nx+1)*dim + dim + i + gridsize_2D;
				counter++;
			}

		}

		// west
		// if ( id < (Nx+1)*(Ny+1)*(Nz)*dim && (id) % ((Nx + 1)*dim) >= dim )
		if ( next_layer && west )
		{

			for(int i = 0 ; i < dim ; i++)
			{
				index[counter + id*max_row_size] = (id - id%dim) - dim + i + gridsize_2D;
				counter++;
			}

		}

	

		// origin
		// if ( id < (Nx+1)*(Ny+1)*(Nz)*dim )
		if ( next_layer )
		{
			for(int i = 0 ; i < dim ; i++)
			{
				index[counter + id*max_row_size] = (id - id%dim) + i + gridsize_2D;
				counter++;
			}
		}



		// east
		// if ( id < (Nx+1)*(Ny+1)*(Nz)*dim && (base_id) % ((Nx*dim) + (base_id/(3*(Nx+1)))*dim*(Nx+1)) != 0 || base_id == 0 )
		if ( base_id == 0 || ( next_layer && east ) )
		{

			for(int i = 0 ; i < dim ; i++)
				{
					index[counter + id*max_row_size] = (id - id%dim) + dim + i + gridsize_2D;
					counter++;
				}
		}




		// north-west
		// if ( id < (Nx+1)*(Ny+1)*(Nz)*dim && (id) % ((Nx + 1)*(Ny + 1)*dim) < (Nx+1)*(Ny)*dim && (id) % ((Nx + 1)*dim) >= dim )
		if ( next_layer && north && west )
		{
			for(int i = 0 ; i < dim ; i++)
			{
				index[counter + id*max_row_size] = (id - id%dim) + (Nx+1)*dim - dim + i + gridsize_2D;
				counter++;
			}

		}

		// north
		// if ( id < (Nx+1)*(Ny+1)*(Nz)*dim && (id) % ((Nx + 1)*(Ny + 1)*dim) < (Nx+1)*(Ny)*dim )
		if ( next_layer && north )
		{
			for(int i = 0 ; i < dim ; i++)
			{
				index[counter + id*max_row_size] = (id - id%dim) + (Nx+1)*dim + i + gridsize_2D;
				counter++;
			}

		}

		// north-east
		// if ( id < (Nx+1)*(Ny+1)*(Nz)*dim && (id) % ((Nx + 1)*(Ny + 1)*dim) < (Nx+1)*(Ny)*dim && (base_id) % ((Nx*dim) + (base_id/(3*(Nx+1)))*dim*(Nx+1)) != 0 || base_id == 0 )
		if ( base_id == 0 || (next_layer && north && east ) )
		{
			for(int i = 0 ; i < dim ; i++)
			{
				index[counter + id*max_row_size] = (id - id%dim) + (Nx+1)*dim + dim + i + gridsize_2D;
				counter++;
			}

		}









	for ( int i = counter ; i < max_row_size; i++)
	{
		index[i + id*max_row_size] = num_rows;
	}









	// if ( id == 0 )
	// {
	// 	for ( int i = 0 ; i < max_row_size ; i++ )
	// 		printf( "%lu ", index[i + id*max_row_size] );

	// 	printf("\n");
	// }





}

}

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
			p_index[counter + id*p_max_row_size] = coarse_node_index*dim + id%dim;
			p_value[counter + id*p_max_row_size] = 1;
			counter++;
		}

		else
		{
			// south-west
			if ( south && condition1 && !condition2 && west ) 
			{
				size_t south_west_fine_node = (node_index - (Nx+1) - 1);
				size_t south_west_coarse_node = getCoarseNode_GPU(south_west_fine_node, Nx, Ny, 0, dim);
				p_index[counter + id*p_max_row_size] = south_west_coarse_node*dim + id%dim ;
				p_value[counter + id*p_max_row_size] = 0.25 ;
				counter++;
			}

			// south
			if ( south && !condition1 && !condition2 )
			{
				size_t south_fine_node = (node_index - (Nx+1) );
				size_t south_coarse_node = getCoarseNode_GPU(south_fine_node, Nx, Ny, 0, dim);
				p_index[counter + id*p_max_row_size] = south_coarse_node*dim + id%dim ;
				p_value[counter + id*p_max_row_size] = 0.5 ;
				counter++;
			}

			// south-east
			if ( south && condition1 && !condition2 && east ) 
			{
				size_t south_east_fine_node = (node_index - (Nx+1) + 1);
				size_t south_east_coarse_node = getCoarseNode_GPU(south_east_fine_node, Nx, Ny, 0, dim);
				p_index[counter + id*p_max_row_size] = south_east_coarse_node*dim + id%dim ;
				p_value[counter + id*p_max_row_size] = 0.25 ;
				counter++;
			}

			// west
			if ( west && condition2 )
			{
				size_t west_fine_node = (node_index - 1);
				size_t west_coarse_node = getCoarseNode_GPU(west_fine_node, Nx, Ny, 0, dim);
				p_index[counter + id*p_max_row_size] = west_coarse_node*dim + id%dim ;
				p_value[counter + id*p_max_row_size] = 0.5 ;
				counter++;
			}

			// east
			if ( east && condition2 )
			{
				size_t east_fine_node = (node_index + 1);
				size_t east_coarse_node = getCoarseNode_GPU(east_fine_node, Nx, Ny, 0, dim);
				p_index[counter + id*p_max_row_size] = east_coarse_node*dim + id%dim ;
				p_value[counter + id*p_max_row_size] = 0.5 ;
				counter++;
			}

			// north-west
			if ( north && condition1 && !condition2 && west )
			{
				size_t north_west_fine_node = (node_index + (Nx+1) - 1);
				size_t north_west_coarse_node = getCoarseNode_GPU(north_west_fine_node, Nx, Ny, 0, dim);
				p_index[counter + id*p_max_row_size] = north_west_coarse_node*dim + id%dim ;
				p_value[counter + id*p_max_row_size] = 0.25 ;
				counter++;
			}

			// north
			if ( north && !condition1 && !condition2 )
			{
				size_t north_fine_node = (node_index + (Nx+1) );
				size_t north_coarse_node = getCoarseNode_GPU(north_fine_node, Nx, Ny, 0, dim);
				p_index[counter + id*p_max_row_size] = north_coarse_node*dim + id%dim ;
				p_value[counter + id*p_max_row_size] = 0.5 ;
				counter++;
			}

			// north-east
			if ( north && condition1 && !condition2 && east ) 
			{
				size_t north_east_fine_node = (node_index + (Nx+1) + 1);
				size_t north_east_coarse_node = getCoarseNode_GPU(north_east_fine_node, Nx, Ny, 0, dim);
				p_index[counter + id*p_max_row_size] = north_east_coarse_node*dim + id%dim ;
				p_value[counter + id*p_max_row_size] = 0.25 ;
				counter++;
			}

		}



		for ( int i = counter ; i < p_max_row_size; i++)
		{
			p_index[i + id*p_max_row_size] = num_cols;
		}

	}
}


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
		size_t numNodes = (numNodes2D)*(Nz+1);


		// if node is even numbered
		bool condition1 = ( node_index % 2 == 0 );

		// if node exists in the coarse grid (x-y-plane)
		bool condition2 = ( node_index % ((Nx+1)*2) < (Nx+1) && node_index < (Nx+1)*(Ny+1) );
		// bool condition2 = ( node_index % ((Nx+1)*(Ny+1)) < (Nx + 1) || node_index % ((Nx+1)*(Ny+1)) >= 2*(Nx + 1));

		// if node exists in the coarse grid (y-z-plane)
		bool condition3 = ( node_index % ((Nx+1)*(Ny+1)*2) < (Nx+1)*(Ny+1) );

		bool condition4 = ( node_index % ((Nx+1)*(Ny+1)*2) < (Nx+1) );
		
		bool condition5 = ( (id_2D/dim) % ((Nx+1)*2) < (Nx+1) );
		bool condition6 = ( node_index % (numNodes2D*2) < (Nx+1)*(Ny+1) );




		bool south = ( id_2D >= (Nx + 1)*dim );
		bool west  = ( (id) % ((Nx + 1)*dim) >= dim );
		bool east  = ( (base_id) % ((Nx*dim) + (base_id/(2*(Nx+1)))*dim*(Nx+1)) != 0 );
		bool north = ( id_2D < (Nx+1)*(Ny)*dim );
		bool previous = ( coarse_node_index >= (Nx_+1)*(Ny_+1) );
		bool next = ( coarse_node_index + (Nx_+1)*(Ny_+1) <= (Nx_+1)*(Ny_+1) );

		

		// node-traversal operations
		size_t previous_ = -numNodes2D;
		size_t next_ = numNodes2D;
		size_t west_ = -1;
		size_t east_ = 1;
		size_t south_ = -(Nx+1);
		size_t north_ = +(Nx+1);

		// if there exists a coarse node in the same location
		if ( getFineNode_GPU(coarse_node_index, Nx_, Ny_, Nz_, dim) == node_index )
		{
			p_index[counter + id*p_max_row_size] = coarse_node_index*dim + id%dim;
			p_value[counter + id*p_max_row_size] = 1;
			counter++;
		}


		// diagonals
		else if ( !condition1 && !condition5 && !condition6 )
		{
			size_t fine_node;
			size_t coarse_node;

			// if ( id == 87 )
			// printf("%lu\n", node_index - numNodes2D - (Nx+1) - 1  );

			// previous-south-west
			fine_node = (node_index - numNodes2D - (Nx+1) - 1 );
			coarse_node = getCoarseNode3D_GPU(fine_node, Nx, Ny, Nz);
			p_index[counter + id*p_max_row_size] = coarse_node*dim + id%dim ;
			p_value[counter + id*p_max_row_size] = 0.125 ;
			counter++;



			// previous-south-east
			fine_node = (node_index - numNodes2D - (Nx+1) + 1 );
			coarse_node = getCoarseNode3D_GPU(fine_node, Nx, Ny, Nz);
			p_index[counter + id*p_max_row_size] = coarse_node*dim + id%dim ;
			p_value[counter + id*p_max_row_size] = 0.125 ;
			counter++;

			// previous-north-west
			fine_node = (node_index - numNodes2D + (Nx+1) - 1 );
			coarse_node = getCoarseNode3D_GPU(fine_node, Nx, Ny, Nz);
			p_index[counter + id*p_max_row_size] = coarse_node*dim + id%dim ;
			p_value[counter + id*p_max_row_size] = 0.125 ;
			counter++; 

			// previous-north-east
			fine_node = (node_index - numNodes2D + (Nx+1) + 1 );
			coarse_node = getCoarseNode3D_GPU(fine_node, Nx, Ny, Nz);
			p_index[counter + id*p_max_row_size] = coarse_node*dim + id%dim ;
			p_value[counter + id*p_max_row_size] = 0.125 ;
			counter++; 

			// next-south-west
			fine_node = (node_index + numNodes2D - (Nx+1) - 1 );
			coarse_node = getCoarseNode3D_GPU(fine_node, Nx, Ny, Nz);
			p_index[counter + id*p_max_row_size] = coarse_node*dim + id%dim ;
			p_value[counter + id*p_max_row_size] = 0.125 ;
			counter++;

			// next-south-east
			fine_node = (node_index + numNodes2D - (Nx+1) + 1);
			coarse_node = getCoarseNode3D_GPU(fine_node, Nx, Ny, Nz);
			p_index[counter + id*p_max_row_size] = coarse_node*dim + id%dim ;
			p_value[counter + id*p_max_row_size] = 0.125 ;
			counter++;

			// next-north-west
			fine_node = (node_index + numNodes2D + (Nx+1) - 1 );
			coarse_node = getCoarseNode3D_GPU(fine_node, Nx, Ny, Nz);
			p_index[counter + id*p_max_row_size] = coarse_node*dim + id%dim ;
			p_value[counter + id*p_max_row_size] = 0.125 ;
			counter++; 

			// next-north-east
			fine_node = (node_index + numNodes2D + (Nx+1) + 1);
			coarse_node = getCoarseNode3D_GPU(fine_node, Nx, Ny, Nz);
			p_index[counter + id*p_max_row_size] = coarse_node*dim + id%dim ;
			p_value[counter + id*p_max_row_size] = 0.125 ;
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
			p_index[counter + id*p_max_row_size] = coarse_node*dim + id%dim ;
			p_value[counter + id*p_max_row_size] = 0.25 ;
			counter++;

			// previous-east
			fine_node = (node_index - (Nx+1)*(Ny+1) + 1);
			coarse_node = getCoarseNode3D_GPU(fine_node, Nx, Ny, Nz);
			p_index[counter + id*p_max_row_size] = coarse_node*dim + id%dim ;
			p_value[counter + id*p_max_row_size] = 0.25 ;
			counter++;

			// next-west
			fine_node = (node_index + (Nx+1)*(Ny+1) - 1);
			coarse_node = getCoarseNode3D_GPU(fine_node, Nx, Ny, Nz);
			p_index[counter + id*p_max_row_size] = coarse_node*dim + id%dim ;
			p_value[counter + id*p_max_row_size] = 0.25 ;
			counter++;

			// next-east
			fine_node = (node_index + (Nx+1)*(Ny+1) + 1);
			coarse_node = getCoarseNode3D_GPU(fine_node, Nx, Ny, Nz);
			p_index[counter + id*p_max_row_size] = coarse_node*dim + id%dim ;
			p_value[counter + id*p_max_row_size] = 0.25 ;
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
			p_index[counter + id*p_max_row_size] = coarse_node*dim + id%dim ;
			p_value[counter + id*p_max_row_size] = 0.25 ;
			counter++;

			// south-east
			fine_node = (node_index - (Nx+1) + 1);
			coarse_node = getCoarseNode3D_GPU(fine_node, Nx, Ny, Nz);
			p_index[counter + id*p_max_row_size] = coarse_node*dim + id%dim ;
			p_value[counter + id*p_max_row_size] = 0.25 ;
			counter++;

			// north-east
			fine_node = (node_index + (Nx+1) - 1);
			coarse_node = getCoarseNode3D_GPU(fine_node, Nx, Ny, Nz);
			p_index[counter + id*p_max_row_size] = coarse_node*dim + id%dim ;
			p_value[counter + id*p_max_row_size] = 0.25 ;
			counter++;

			// north-east
			fine_node = (node_index + (Nx+1) + 1);
			coarse_node = getCoarseNode3D_GPU(fine_node, Nx, Ny, Nz);
			p_index[counter + id*p_max_row_size] = coarse_node*dim + id%dim ;
			p_value[counter + id*p_max_row_size] = 0.25 ;
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
			p_index[counter + id*p_max_row_size] = coarse_node*dim + id%dim ;
			p_value[counter + id*p_max_row_size] = 0.25 ;
			counter++;

			// previous-north
			fine_node = (node_index - (Nx+1)*(Ny+1) + (Nx+1) );
			coarse_node = getCoarseNode3D_GPU(fine_node, Nx, Ny, Nz);
			p_index[counter + id*p_max_row_size] = coarse_node*dim + id%dim ;
			p_value[counter + id*p_max_row_size] = 0.25 ;
			counter++;

			// next-south
			fine_node = (node_index + (Nx+1)*(Ny+1) - (Nx+1) );
			coarse_node = getCoarseNode3D_GPU(fine_node, Nx, Ny, Nz);
			p_index[counter + id*p_max_row_size] = coarse_node*dim + id%dim ;
			p_value[counter + id*p_max_row_size] = 0.25 ;
			counter++;

			// next-north
			fine_node = (node_index + (Nx+1)*(Ny+1) + (Nx+1) );
			coarse_node = getCoarseNode3D_GPU(fine_node, Nx, Ny, Nz);
			p_index[counter + id*p_max_row_size] = coarse_node*dim + id%dim ;
			p_value[counter + id*p_max_row_size] = 0.25 ;
			counter++;
		}

		else
		{		
			// if ( id == 171 )
			// 	printf("%d\n", condition5 );

			// DONE:
			// previous-origin
			if ( !condition1 && condition5 && !condition6 )
			{
				// printf("%lu\n", node_index*dim );
				size_t fine_node = (node_index - (Nx+1)*(Ny+1));
				size_t coarse_node = getCoarseNode3D_GPU(fine_node, Nx, Ny, Nz);
				p_index[counter + id*p_max_row_size] = coarse_node*dim + id%dim ;
				p_value[counter + id*p_max_row_size] = 0.5 ;
				counter++;
			}

			// DONE:
			// next-origin
			if ( !condition1 && condition5 && !condition6 )
			{
				// printf("%lu\n", node_index*dim );
				size_t fine_node = (node_index + (Nx+1)*(Ny+1));
				size_t coarse_node = getCoarseNode3D_GPU(fine_node, Nx, Ny, Nz);
				p_index[counter + id*p_max_row_size] = coarse_node*dim + id%dim ;
				p_value[counter + id*p_max_row_size] = 0.5 ;
				counter++;
			}

			// DONE:
			// south
			if ( !condition1 && !condition5 && condition6 )
			{
				
				size_t fine_node = (node_index - (Nx+1));
				size_t coarse_node = getCoarseNode3D_GPU(fine_node, Nx, Ny, Nz);
				p_index[counter + id*p_max_row_size] = coarse_node*dim + id%dim ;
				p_value[counter + id*p_max_row_size] = 0.5 ;
				counter++;
			}

			// DONE:
			// west
			if ( !condition1 && condition5 && condition6 )
			{
				// printf("%lu\n", node_index*3 );
				size_t fine_node = (node_index - 1);
				size_t coarse_node = getCoarseNode3D_GPU(fine_node, Nx, Ny, Nz);
				p_index[counter + id*p_max_row_size] = coarse_node*dim + id%dim ;
				p_value[counter + id*p_max_row_size] = 0.5 ;
				counter++;
			}


			// DONE:
			// east
			if ( !condition1 && condition5 && condition6 )
			{
				
				size_t fine_node = (node_index + 1);
				size_t coarse_node = getCoarseNode3D_GPU(fine_node, Nx, Ny, Nz);
				p_index[counter + id*p_max_row_size] = coarse_node*dim + id%dim ;
				p_value[counter + id*p_max_row_size] = 0.5 ;
				counter++;
			}

			// DONE:
			// north
			if ( !condition1 && !condition5 && condition6 )
			{
				
				size_t fine_node = (node_index + (Nx+1));
				size_t coarse_node = getCoarseNode3D_GPU(fine_node, Nx, Ny, Nz);
				p_index[counter + id*p_max_row_size] = coarse_node*dim + id%dim ;
				p_value[counter + id*p_max_row_size] = 0.5 ;
				counter++;
			}

		}

		for ( int i = counter ; i < p_max_row_size; i++)
			{
				p_index[i + id*p_max_row_size] = num_cols;
			}

		// if (id == 84)
		// {
		// 	for ( int i = 0 ; i < p_max_row_size; i++)
		// 		printf("%lu ", p_index[i + id*p_max_row_size]);

		// 	printf("\n");

		// 	for ( int i = 0 ; i < p_max_row_size; i++)
		// 		printf("%g ", p_value[i + id*p_max_row_size]);

		// 	printf("\n");
		// }











			// // previous-south-west
			// if ( previous && !condition1 && !condition2 && !condition3 && !condition4 )
			// {
			// 	// printf("%lu\n", node_index);
			// 	size_t fine_node = node_index + previous_ + west_ + south_;
			// 	size_t coarse_node = getCoarseNode3D_GPU(fine_node, Nx, Ny, Nz);
			// 	p_index[counter + id*p_max_row_size] = coarse_node*dim + id%dim ;
			// 	p_value[counter + id*p_max_row_size] = 0.125;
			// 	counter++;
			// }

			// // previous-south
			// if ( previous && condition1 && !condition2 &&!condition3 && !condition4 )
			// {
			// 	// printf("%lu\n", node_index);
				
			// 	size_t fine_node = node_index + previous_ + south_;
			// 	size_t coarse_node = getCoarseNode3D_GPU(fine_node, Nx, Ny, Nz);
			// 	p_index[counter + id*p_max_row_size] = coarse_node*dim + id%dim ;
			// 	p_value[counter + id*p_max_row_size] = 0.25;
			// 	counter++;
			// }

			// // previous-south-east
			// if ( previous && !condition1 && !condition2 && !condition3 && !condition4 )
			// {

			// 	// printf("%lu\n", node_index);
				
			// 	size_t fine_node = node_index + previous_ + south_ + east_;
			// 	size_t coarse_node = getCoarseNode3D_GPU(fine_node, Nx, Ny, Nz);
			// 	p_index[counter + id*p_max_row_size] = coarse_node*dim + id%dim ;
			// 	p_value[counter + id*p_max_row_size] = 0.125;
			// 	counter++;
			// }

			// // previous-west
			// if ( previous && condition1 && condition2 && !condition3 )
			// {
			// 	// printf("%lu\n", node_index);
				
			// 	size_t fine_node = node_index + previous_ + west_;
			// 	size_t coarse_node = getCoarseNode3D_GPU(fine_node, Nx, Ny, Nz);
			// 	p_index[counter + id*p_max_row_size] = coarse_node*dim + id%dim ;
			// 	p_value[counter + id*p_max_row_size] = 0.25;
			// 	counter++;
			// }

			// // previous-origin
			// if ( previous && !condition1 && condition2 && !condition3 )
			// {
			// 	// printf("%lu\n", node_index);
				
			// 	size_t fine_node = node_index + previous_;
			// 	size_t coarse_node = getCoarseNode3D_GPU(fine_node, Nx, Ny, Nz);
			// 	p_index[counter + id*p_max_row_size] = coarse_node*dim + id%dim ;
			// 	p_value[counter + id*p_max_row_size] = 0.5;
			// 	counter++;
			// }

			// // previous-east
			// if ( previous && condition1 && condition2 && !condition3 )
			// {
			// 	// printf("%lu\n", node_index);
				
			// 	size_t fine_node = node_index + previous_ + east_;
			// 	size_t coarse_node = getCoarseNode3D_GPU(fine_node, Nx, Ny, Nz);
			// 	p_index[counter + id*p_max_row_size] = coarse_node*dim + id%dim ;
			// 	p_value[counter + id*p_max_row_size] = 0.25;
			// 	counter++;
			// }

			// // previous-north-west
			// if ( previous && !condition1 && !condition2 && !condition3 && !condition4 )
			// {
			// 	// printf("%lu\n", node_index);
			// 	size_t fine_node = node_index + previous_ + west_ + north_;
			// 	size_t coarse_node = getCoarseNode3D_GPU(fine_node, Nx, Ny, Nz);
			// 	p_index[counter + id*p_max_row_size] = coarse_node*dim + id%dim ;
			// 	p_value[counter + id*p_max_row_size] = 0.125;
			// 	counter++;
			// }

			// // previous-north
			// if ( previous && condition1 && !condition2 &&!condition3 && !condition4 )
			// {
			// 	size_t fine_node = node_index + previous_ + north_;
			// 	size_t coarse_node = getCoarseNode3D_GPU(fine_node, Nx, Ny, Nz);
			// 	p_index[counter + id*p_max_row_size] = coarse_node*dim + id%dim ;
			// 	p_value[counter + id*p_max_row_size] = 0.25;
			// 	counter++;
			// }

			// // previous-north-east
			// if ( previous && !condition1 && !condition2 && !condition3 && !condition4 )
			// {
			// 	// printf("%lu\n", node_index);
			// 	size_t fine_node = node_index + previous_ + east_ + north_;
			// 	size_t coarse_node = getCoarseNode3D_GPU(fine_node, Nx, Ny, Nz);
			// 	p_index[counter + id*p_max_row_size] = coarse_node*dim + id%dim ;
			// 	p_value[counter + id*p_max_row_size] = 0.125;
			// 	counter++;
			// }




			// // south-west
			// if ( south && condition1 && !condition2 && condition3 && west ) 
			// {
			// 	size_t south_west_fine_node = (node_index - (Nx+1) - 1);
			// 	size_t south_west_coarse_node = getCoarseNode3D_GPU(south_west_fine_node, Nx, Ny, Nz);
			// 	p_index[counter + id*p_max_row_size] = south_west_coarse_node*dim + id%dim ;
			// 	p_value[counter + id*p_max_row_size] = 0.25 ;
			// 	counter++;
			// }


			// // south
			// if ( south && !condition1 && !condition2 && condition3)
			// {
			// 	size_t south_fine_node = (node_index - (Nx+1) );
			// 	size_t south_coarse_node = getCoarseNode3D_GPU(south_fine_node, Nx, Ny, Nz);
			// 	p_index[counter + id*p_max_row_size] = south_coarse_node*dim + id%dim ;
			// 	p_value[counter + id*p_max_row_size] = 0.5 ;
			// 	counter++;
			// }

			// // south-east
			// if ( south && condition1 && !condition2 && condition3 && east ) 
			// {
			// 	size_t south_east_fine_node = (node_index - (Nx+1) + 1);
			// 	size_t south_east_coarse_node = getCoarseNode3D_GPU(south_east_fine_node, Nx, Ny, Nz);
			// 	p_index[counter + id*p_max_row_size] = south_east_coarse_node*dim + id%dim ;
			// 	p_value[counter + id*p_max_row_size] = 0.25 ;
			// 	counter++;
			// }

			// // west
			// if ( west && condition2 && condition3 )
			// {
			// 	size_t west_fine_node = (node_index - 1);
			// 	size_t west_coarse_node = getCoarseNode3D_GPU(west_fine_node, Nx, Ny, Nz);
			// 	p_index[counter + id*p_max_row_size] = west_coarse_node*dim + id%dim ;
			// 	p_value[counter + id*p_max_row_size] = 0.5 ;
			// 	counter++;

			// }

			// // east
			// if ( east && condition2 && condition3 && condition5)
			// {
			// 	size_t east_fine_node = (node_index + 1);
			// 	size_t east_coarse_node = getCoarseNode3D_GPU(east_fine_node, Nx, Ny, Nz);
			// 	p_index[counter + id*p_max_row_size] = east_coarse_node*dim + id%dim ;
			// 	p_value[counter + id*p_max_row_size] = 0.5 ;
			// 	counter++;
			// }

			// // north-west
			// if ( north && condition1 && !condition2 && condition3 && west )
			// {
			// 	size_t north_west_fine_node = (node_index + (Nx+1) - 1);
			// 	size_t north_west_coarse_node = getCoarseNode3D_GPU(north_west_fine_node, Nx, Ny, Nz);
			// 	p_index[counter + id*p_max_row_size] = north_west_coarse_node*dim + id%dim ;
			// 	p_value[counter + id*p_max_row_size] = 0.25 ;
			// 	counter++;
			// }


			// // north
			// if ( north && !condition1 && !condition2 && condition3)
			// {
			// 	size_t north_fine_node = (node_index + (Nx+1) );
			// 	size_t north_coarse_node = getCoarseNode3D_GPU(north_fine_node, Nx, Ny, Nz);
			// 	p_index[counter + id*p_max_row_size] = north_coarse_node*dim + id%dim ;
			// 	p_value[counter + id*p_max_row_size] = 0.5 ;
			// 	counter++;
			// }

			// // north-east
			// if ( north && condition1 && !condition2 && condition3 && east ) 
			// {
			// 	size_t north_east_fine_node = (node_index + (Nx+1) + 1);
			// 	size_t north_east_coarse_node = getCoarseNode3D_GPU(north_east_fine_node, Nx, Ny, Nz);
			// 	p_index[counter + id*p_max_row_size] = north_east_coarse_node*dim + id%dim ;
			// 	p_value[counter + id*p_max_row_size] = 0.25 ;
			// 	counter++;
			// }
			



			// // next-south-west
			// if ( next && !condition1 && !condition2 && !condition3 && !condition4 )
			// {
			// 	// printf("%lu\n", node_index);
			// 	size_t fine_node = node_index + next_ + west_ + south_;
			// 	size_t coarse_node = getCoarseNode3D_GPU(fine_node, Nx, Ny, Nz);
			// 	p_index[counter + id*p_max_row_size] = coarse_node*dim + id%dim ;
			// 	p_value[counter + id*p_max_row_size] = 0.125;
			// 	counter++;
			// }

			// // next-south
			// if ( next && condition1 && !condition2 &&!condition3 && !condition4 )
			// {
			// 	// printf("%lu\n", node_index);
				
			// 	size_t fine_node = node_index + next_ + south_;
			// 	size_t coarse_node = getCoarseNode3D_GPU(fine_node, Nx, Ny, Nz);
			// 	p_index[counter + id*p_max_row_size] = coarse_node*dim + id%dim ;
			// 	p_value[counter + id*p_max_row_size] = 0.25;
			// 	counter++;
			// }

			// // next-south-east
			// if ( next && !condition1 && !condition2 && !condition3 && !condition4 )
			// {

			// 	// printf("%lu\n", node_index);
				
			// 	size_t fine_node = node_index + next_ + south_ + east_;
			// 	size_t coarse_node = getCoarseNode3D_GPU(fine_node, Nx, Ny, Nz);
			// 	p_index[counter + id*p_max_row_size] = coarse_node*dim + id%dim ;
			// 	p_value[counter + id*p_max_row_size] = 0.125;
			// 	counter++;
			// }

			// // next-west
			// if ( next && condition1 && condition2 && !condition3 )
			// {
			// 	// printf("%lu\n", node_index);
				
			// 	size_t fine_node = node_index + next_ + west_;
			// 	size_t coarse_node = getCoarseNode3D_GPU(fine_node, Nx, Ny, Nz);
			// 	p_index[counter + id*p_max_row_size] = coarse_node*dim + id%dim ;
			// 	p_value[counter + id*p_max_row_size] = 0.25;
			// 	counter++;
			// }

			// // next-origin
			// if ( next && !condition1 && condition2 && !condition3 )
			// {
			// 	// printf("%lu\n", node_index);
				
			// 	size_t fine_node = node_index + next_;
			// 	size_t coarse_node = getCoarseNode3D_GPU(fine_node, Nx, Ny, Nz);
			// 	p_index[counter + id*p_max_row_size] = coarse_node*dim + id%dim ;
			// 	p_value[counter + id*p_max_row_size] = 0.5;
			// 	counter++;
			// }

			// // next-east
			// if ( next && condition1 && condition2 && !condition3 )
			// {
			// 	// printf("%lu\n", node_index);
				
			// 	size_t fine_node = node_index + next_ + east_;
			// 	size_t coarse_node = getCoarseNode3D_GPU(fine_node, Nx, Ny, Nz);
			// 	p_index[counter + id*p_max_row_size] = coarse_node*dim + id%dim ;
			// 	p_value[counter + id*p_max_row_size] = 0.25;
			// 	counter++;
			// }

			// // next-north-west
			// if ( next && !condition1 && !condition2 && !condition3 && !condition4 )
			// {
			// 	// printf("%lu\n", node_index);
			// 	size_t fine_node = node_index + next_ + west_ + north_;
			// 	size_t coarse_node = getCoarseNode3D_GPU(fine_node, Nx, Ny, Nz);
			// 	p_index[counter + id*p_max_row_size] = coarse_node*dim + id%dim ;
			// 	p_value[counter + id*p_max_row_size] = 0.125;
			// 	counter++;
			// }

			// // next-north
			// if ( next && condition1 && !condition2 &&!condition3 && !condition4 )
			// {
			// 	size_t fine_node = node_index + next_ + north_;
			// 	size_t coarse_node = getCoarseNode3D_GPU(fine_node, Nx, Ny, Nz);
			// 	p_index[counter + id*p_max_row_size] = coarse_node*dim + id%dim ;
			// 	p_value[counter + id*p_max_row_size] = 0.25;
			// 	counter++;
			// }

			// // next-north-east
			// if ( next && !condition1 && !condition2 && !condition3 && !condition4 )
			// {
			// 	// printf("%lu\n", node_index);
			// 	size_t fine_node = node_index + next_ + east_ + north_;
			// 	size_t coarse_node = getCoarseNode3D_GPU(fine_node, Nx, Ny, Nz);
			// 	p_index[counter + id*p_max_row_size] = coarse_node*dim + id%dim ;
			// 	p_value[counter + id*p_max_row_size] = 0.125;
			// 	counter++;
			// }


		


		

	}
}

__device__ int getCoarseNode_GPU(size_t index, size_t Nx, size_t Ny, size_t Nz, size_t dim)
{
	// get coarse grid dimensions
	size_t Nx_ = Nx / 2;
	size_t Ny_ = Ny / 2;
	size_t Nz_ = Nz / 2;

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
	size_t Nz_ = Nz / 2;

	size_t gridsize2D = (Nx+1)*(Ny+1);
	size_t gridsize2D_ = (Nx_+1)*(Ny_+1);

	// if node is even numbered
	bool condition1 = ( index % 2 == 0 );

	// if node exists in the coarse grid (x-y-plane)
	bool condition2 = ( index % ((Nx+1)*2) < (Nx + 1) );

	// if node exists in the coarse grid (y-z-plane)
	bool condition3 = ( index % ((Nx+1)*(Ny+1)*2) < (Nx+1)*(Ny+1) );

	// printf("aps = %d\n", ((Nx+1)*2)   );

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


// __global__ void fillIndexVectorProl2D_GPU(size_t* p_index, size_t Nx, size_t Ny, size_t p_max_row_size, size_t num_rows, size_t num_cols)
// {
// 	unsigned int id = threadIdx.x + blockIdx.x*blockDim.x;

// 	if ( id < num_rows )
// 	{
// 		int counter = 0;
// 		int dim = 2;	

// 		// coarse grid
// 		size_t Nx_ = Nx / 2;
// 		size_t Ny_ = Ny / 2;

// 		size_t base_id = (id - id%dim);
// 		size_t node_index = base_id / dim;
// 		int coarse_node_index = getCoarseNode_GPU(node_index, Nx, Ny, 0, dim);
		
// 		// if node is even numbered
// 		bool condition1 = (node_index % 2 == 0 );

// 		// if node exists in the coarse grid
// 		bool condition2 = ( node_index % ((Nx+1)*2) < (Nx + 1) );

// 		bool south = ( id  >= (Nx + 1)*dim );
// 		bool west  = ( (id) % ((Nx + 1)*dim) >= dim );
// 		bool east  = ( (base_id) % ((Nx*dim) + (base_id/(2*(Nx+1)))*dim*(Nx+1)) != 0 );
// 		bool north = ( id < (Nx+1)*(Ny)*dim );


// 		// if there exists a coarse node in the same location
// 		if ( getFineNode_GPU(coarse_node_index, Nx_, Ny_, 0, dim) == node_index )
// 		{
// 			p_index[counter + id*p_max_row_size] = coarse_node_index*dim + id%dim;
// 			counter++;
// 		}

// 		else
// 		{
// 			// south-west
// 			if ( south && condition1 && !condition2 && west ) 
// 			{
// 				size_t south_west_fine_node = (node_index - (Nx+1) - 1);
// 				size_t south_west_coarse_node = getCoarseNode_GPU(south_west_fine_node, Nx, Ny, 0, dim);
// 				p_index[counter + id*p_max_row_size] = south_west_coarse_node*dim + id%dim ;
// 				counter++;
// 			}

// 			// south
// 			if ( south && !condition1 && !condition2 )
// 			{
// 				size_t south_fine_node = (node_index - (Nx+1) );
// 				size_t south_coarse_node = getCoarseNode_GPU(south_fine_node, Nx, Ny, 0, dim);
// 				p_index[counter + id*p_max_row_size] = south_coarse_node*dim + id%dim ;
// 				counter++;
// 			}

// 			// south-east
// 			if ( south && condition1 && !condition2 && east ) 
// 			{
// 				size_t south_east_fine_node = (node_index - (Nx+1) + 1);
// 				size_t south_east_coarse_node = getCoarseNode_GPU(south_east_fine_node, Nx, Ny, 0, dim);
// 				p_index[counter + id*p_max_row_size] = south_east_coarse_node*dim + id%dim ;
// 				counter++;
// 			}

// 			// west
// 			if ( west && condition2 )
// 			{
// 				size_t west_fine_node = (node_index - 1);
// 				size_t west_coarse_node = getCoarseNode_GPU(west_fine_node, Nx, Ny, 0, dim);
// 				p_index[counter + id*p_max_row_size] = west_coarse_node*dim + id%dim ;
// 				counter++;
// 			}

// 			// east
// 			if ( east && condition2 )
// 			{
// 				size_t east_fine_node = (node_index + 1);
// 				size_t east_coarse_node = getCoarseNode_GPU(east_fine_node, Nx, Ny, 0, dim);
// 				p_index[counter + id*p_max_row_size] = east_coarse_node*dim + id%dim ;
// 				counter++;
// 			}

// 			// north-west
// 			if ( north && condition1 && !condition2 && west )
// 			{
// 				size_t north_west_fine_node = (node_index + (Nx+1) - 1);
// 				size_t north_west_coarse_node = getCoarseNode_GPU(north_west_fine_node, Nx, Ny, 0, dim);
// 				p_index[counter + id*p_max_row_size] = north_west_coarse_node*dim + id%dim ;
// 				counter++;
// 			}

// 			// north
// 			if ( north && !condition1 && !condition2 )
// 			{
// 				size_t north_fine_node = (node_index + (Nx+1) );
// 				size_t north_coarse_node = getCoarseNode_GPU(north_fine_node, Nx, Ny, 0, dim);
// 				p_index[counter + id*p_max_row_size] = north_coarse_node*dim + id%dim ;
// 				counter++;
// 			}

// 			// north-east
// 			if ( north && condition1 && !condition2 && east ) 
// 			{
// 				size_t north_east_fine_node = (node_index + (Nx+1) + 1);
// 				size_t north_east_coarse_node = getCoarseNode_GPU(north_east_fine_node, Nx, Ny, 0, dim);
// 				p_index[counter + id*p_max_row_size] = north_east_coarse_node*dim + id%dim ;
// 				counter++;
// 			}

// 		}


// 		// else if ( coarse_node_index == -1 )
// 		// {
// 		// 	print
// 		// }
// 		// bool origin = ( id )
		
// 		// //
// 		// if ( id == getFineNode_GPU(id, Nx, Ny, 0, dim) )
// 		// {
// 		// 	p_index[counter + id*p_max_row_size] = getFineNode_GPU(id, Nx, Ny, 0, dim);
// 		// 	counter++;
// 		// }

// 		// else
// 		// {

// 		// }



// 		for ( int i = counter ; i < p_max_row_size; i++)
// 		{
// 			p_index[i + id*p_max_row_size] = num_cols;
// 		}

// 	}
// }









__global__ void fillIndexVectorRest2D_GPU(size_t* r_index, size_t Nx, size_t Ny, size_t r_max_row_size, size_t num_rows, size_t num_cols)
{
	unsigned int id = threadIdx.x + blockIdx.x*blockDim.x;

	int counter = 0;
	int dim = 2;	

	if ( id < num_rows )
	{
	size_t coarse_node_index = id / dim;
	size_t fine_id = getFineNode_GPU(id, Nx, Ny, 0, dim);
	size_t base_id = (id - id%dim);


	// all on fine grid
	// base : dim*getFineNode_GPU(coarse_node_index, Nx, Ny, 0, dim) = (id - id%dim)
	// s : - ((Nx)*dim + 1)*2 = - (Nx+1)*dim
	// w : - dim


	// south-west
	if ( id  >= (Nx + 1)*dim && (id) % ((Nx + 1)*dim) >= dim )
	{
		r_index[counter + id*r_max_row_size] = dim*getFineNode_GPU(coarse_node_index, Nx, Ny, 0, dim) - ((Nx)*dim + 1)*2 - dim + id%dim;
		counter++;
	}

	// south
	if ( id  >= (Nx + 1)*dim )
	{
		r_index[counter + id*r_max_row_size] = dim*getFineNode_GPU(coarse_node_index, Nx, Ny, 0, dim) - ((Nx)*dim + 1)*2 + id%dim;
		counter++;
	}

	// south-east
	if ( id  >= (Nx + 1)*dim && (base_id) % ((Nx*dim) + (base_id/(2*(Nx+1)))*dim*(Nx+1)) != 0 )
	{
		r_index[counter + id*r_max_row_size] = dim*getFineNode_GPU(coarse_node_index, Nx, Ny, 0, dim) - ((Nx)*dim + 1)*2 + dim + id%dim;
		counter++;
	}

	// west
	if ( (id) % ((Nx + 1)*dim) >= dim )
	{
		r_index[counter + id*r_max_row_size] = dim*getFineNode_GPU(coarse_node_index, Nx, Ny, 0, dim) - dim + id%dim;
		counter++;
	}

	// origin
		r_index[counter + id*r_max_row_size] = dim*getFineNode_GPU(coarse_node_index, Nx, Ny, 0, dim) + id%dim;
		counter++;

	// east
	if ( base_id == 0 || (base_id) % ((Nx*dim) + (base_id/(2*(Nx+1)))*dim*(Nx+1)) != 0 )
	{
		r_index[counter + id*r_max_row_size] = dim*getFineNode_GPU(coarse_node_index, Nx, Ny, 0, dim) + dim + id%dim;
		counter++;
	}

	// north-west
	if ( id < (Nx+1)*(Ny)*dim && (id) % ((Nx + 1)*dim) >= dim )
	{
		r_index[counter + id*r_max_row_size] = dim*getFineNode_GPU(coarse_node_index, Nx, Ny, 0, dim) + ((Nx)*dim + 1)*2 - dim + id%dim;
		counter++;
	}

	// north
	if ( id < (Nx+1)*(Ny)*dim )
	{
		r_index[counter + id*r_max_row_size] = dim*getFineNode_GPU(coarse_node_index, Nx, Ny, 0, dim) + ((Nx)*dim + 1)*2 + id%dim;
		counter++;
	}

	// north-east
	if ( base_id == 0 || id < (Nx+1)*(Ny)*dim && (base_id) % ((Nx*dim) + (base_id/(2*(Nx+1)))*dim*(Nx+1)) != 0 )
	{
		r_index[counter + id*r_max_row_size] = dim*getFineNode_GPU(coarse_node_index, Nx, Ny, 0, dim) + ((Nx)*dim + 1)*2 + dim + id%dim;
		counter++;
	}

	for ( int i = counter ; i < r_max_row_size; i++)
	{
		r_index[i + id*r_max_row_size] = num_cols;
	}

	}
}

__global__ void fillIndexVectorRest3D_GPU(size_t* r_index, size_t Nx, size_t Ny, size_t Nz, size_t r_max_row_size, size_t num_rows, size_t num_cols)
{
	unsigned int id = threadIdx.x + blockIdx.x*blockDim.x;

	int counter = 0;
	int dim = 3;	

	if ( id < num_rows )
	{
	size_t coarse_node_index = id / dim;
	size_t fine_id = getFineNode_GPU(id, Nx, Ny, 0, dim);
	size_t base_id = (id - id%dim);
	size_t baseid_2D = (id) % ((Nx + 1)*(Ny + 1)*dim);


	// all on fine grid
	// base : dim*getFineNode_GPU(coarse_node_index, Nx, Ny, 0, dim) = (id - id%dim)
	
	// w : - dim
	// n : ((Nx)*2 + 1)*3
	// s : - ((Nx)*2 + 1)*3
	// previous layer
	// id >= (Nx+1)*(Ny+1)


	// TODO: take above index's || base ...
	//// previous layer

		// south-west
		if ( id >= (Nx+1)*(Ny+1)*dim && (id) % ((Nx + 1)*(Ny + 1)*dim) >= (Nx + 1)*dim && (id) % ((Nx + 1)*dim) >= dim )
		{
			r_index[counter + id*r_max_row_size] = dim*getFineNode_GPU(coarse_node_index, Nx, Ny, Nz, dim) - ((Nx)*2 + 1)*3 - dim + id%dim - (2*Nx+1)*(2*Ny+1)*3;
			counter++;
		}

		// south
		if ( id >= (Nx+1)*(Ny+1)*dim && (id) % ((Nx + 1)*(Ny + 1)*dim) >= (Nx + 1)*dim )
		{
			r_index[counter + id*r_max_row_size] = dim*getFineNode_GPU(coarse_node_index, Nx, Ny, Nz, dim) - ((Nx)*2 + 1)*3 + id%dim - (2*Nx+1)*(2*Ny+1)*3;
			counter++;
		}

		// south-east
		if ( id >= (Nx+1)*(Ny+1)*dim && (id) % ((Nx + 1)*(Ny + 1)*dim) >= (Nx + 1)*dim && (base_id) % ((Nx*dim) + (base_id/(3*(Nx+1)))*dim*(Nx+1)) != 0 )
		{	
			r_index[counter + id*r_max_row_size] = dim*getFineNode_GPU(coarse_node_index, Nx, Ny, Nz, dim) - ((Nx)*2 + 1)*3 + dim + id%dim - (2*Nx+1)*(2*Ny+1)*3;
			counter++;
		}

		// west
		if ( id >= (Nx+1)*(Ny+1)*dim && (id) % ((Nx + 1)*dim) >= dim )
		{
			r_index[counter + id*r_max_row_size] = dim*getFineNode_GPU(coarse_node_index, Nx, Ny, Nz, dim) - dim + id%dim - (2*Nx+1)*(2*Ny+1)*3;
			counter++;
		}

		// origin
		if ( id >= (Nx+1)*(Ny+1)*dim && id != 0)
		{
				r_index[counter + id*r_max_row_size] = dim*getFineNode_GPU(coarse_node_index, Nx, Ny, Nz, dim) + id%dim - (2*Nx+1)*(2*Ny+1)*3;
				counter++;

		}

		// east
		if ( id >= (Nx+1)*(Ny+1)*dim && (base_id) % ((Nx*dim) + (base_id/(3*(Nx+1)))*dim*(Nx+1)) != 0 )
		{
			r_index[counter + id*r_max_row_size] = dim*getFineNode_GPU(coarse_node_index, Nx, Ny, Nz, dim) + dim + id%dim - (2*Nx+1)*(2*Ny+1)*3;
			counter++;
		}

		// north-west
		if ( id >= (Nx+1)*(Ny+1)*dim && (id) % ((Nx + 1)*(Ny + 1)*dim) < (Nx+1)*(Ny)*dim && (id) % ((Nx + 1)*dim) >= dim )
		{
			r_index[counter + id*r_max_row_size] = dim*getFineNode_GPU(coarse_node_index, Nx, Ny, Nz, dim) + ((Nx)*2 + 1)*3 - dim + id%dim - (2*Nx+1)*(2*Ny+1)*3;
			counter++;
		}

		// north
		if ( id >= (Nx+1)*(Ny+1)*dim && (id) % ((Nx + 1)*(Ny + 1)*dim) < (Nx+1)*(Ny)*dim )
		{
			r_index[counter + id*r_max_row_size] = dim*getFineNode_GPU(coarse_node_index, Nx, Ny, Nz, dim) + ((Nx)*2 + 1)*3 + id%dim - (2*Nx+1)*(2*Ny+1)*3;
			counter++;
		}

		// north-east
		if ( id >= (Nx+1)*(Ny+1)*dim && (id) % ((Nx + 1)*(Ny + 1)*dim) < (Nx+1)*(Ny)*dim && (base_id) % ((Nx*dim) + (base_id/(3*(Nx+1)))*dim*(Nx+1)) != 0 )
		{
			r_index[counter + id*r_max_row_size] = dim*getFineNode_GPU(coarse_node_index, Nx, Ny, Nz, dim) + ((Nx)*2 + 1)*3 + dim + id%dim - (2*Nx+1)*(2*Ny+1)*3;
			counter++;
		}










	//// current layer
		
		// south-west
		if ( (id) % ((Nx + 1)*(Ny + 1)*dim)  >= (Nx + 1)*dim && (id) % ((Nx + 1)*dim) >= dim )
		{
			r_index[counter + id*r_max_row_size] = dim*getFineNode_GPU(coarse_node_index, Nx, Ny, Nz, dim) - ((Nx)*2 + 1)*3 - dim + id%dim;
			counter++;
		}

		// south
		if ( (id) % ((Nx + 1)*(Ny + 1)*dim)  >= (Nx + 1)*dim )
		{
			r_index[counter + id*r_max_row_size] = dim*getFineNode_GPU(coarse_node_index, Nx, Ny, Nz, dim) - ((Nx)*2 + 1)*3 + id%dim;
			counter++;
		}

		// south-east
		if ( (id) % ((Nx + 1)*(Ny + 1)*dim)  >= (Nx + 1)*dim && (base_id) % ((Nx*dim) + (base_id/(3*(Nx+1)))*dim*(Nx+1)) != 0 )
		{	
			r_index[counter + id*r_max_row_size] = dim*getFineNode_GPU(coarse_node_index, Nx, Ny, Nz, dim) - ((Nx)*2 + 1)*3 + dim + id%dim;
			counter++;
		}

		// west
		if ( (id) % ((Nx + 1)*dim) >= dim )
		{
			r_index[counter + id*r_max_row_size] = dim*getFineNode_GPU(coarse_node_index, Nx, Ny, Nz, dim) - dim + id%dim;
			counter++;
		}

		// origin
			r_index[counter + id*r_max_row_size] = dim*getFineNode_GPU(coarse_node_index, Nx, Ny, Nz, dim) + id%dim;
			counter++;

		// east
		if ( base_id == 0 || (base_id) % ((Nx*dim) + (base_id/(3*(Nx+1)))*dim*(Nx+1)) != 0 )
		{
			r_index[counter + id*r_max_row_size] = dim*getFineNode_GPU(coarse_node_index, Nx, Ny, Nz, dim) + dim + id%dim;
			counter++;
		}

		// north-west
		if ( (id) % ((Nx + 1)*(Ny + 1)*dim) < (Nx+1)*(Ny)*dim && (id) % ((Nx + 1)*dim) >= dim )
		{
			r_index[counter + id*r_max_row_size] = dim*getFineNode_GPU(coarse_node_index, Nx, Ny, Nz, dim) + ((Nx)*2 + 1)*3 - dim + id%dim;
			counter++;
		}

		// north
		if ( (id) % ((Nx + 1)*(Ny + 1)*dim) < (Nx+1)*(Ny)*dim )
		{
			r_index[counter + id*r_max_row_size] = dim*getFineNode_GPU(coarse_node_index, Nx, Ny, Nz, dim) + ((Nx)*2 + 1)*3 + id%dim;
			counter++;
		}

		// north-east
		if ( base_id == 0 || (id) % ((Nx + 1)*(Ny + 1)*dim) < (Nx+1)*(Ny)*dim && (base_id) % ((Nx*dim) + (base_id/(3*(Nx+1)))*dim*(Nx+1)) != 0 )
		{
			r_index[counter + id*r_max_row_size] = dim*getFineNode_GPU(coarse_node_index, Nx, Ny, Nz, dim) + ((Nx)*2 + 1)*3 + dim + id%dim;
			counter++;
		}


	//// next layer

		
		// south-west
		if ( id < (Nx+1)*(Ny+1)*(Nz)*dim && (id) % ((Nx + 1)*(Ny + 1)*dim) >= (Nx + 1)*dim && (id) % ((Nx + 1)*dim) >= dim )
		{
			r_index[counter + id*r_max_row_size] = dim*getFineNode_GPU(coarse_node_index, Nx, Ny, Nz, dim) - ((Nx)*2 + 1)*3 - dim + id%dim + (2*Nx+1)*(2*Ny+1)*3;
			counter++;
		}

		// south
		if ( id < (Nx+1)*(Ny+1)*(Nz)*dim && (id) % ((Nx + 1)*(Ny + 1)*dim) >= (Nx + 1)*dim )
		{
			r_index[counter + id*r_max_row_size] = dim*getFineNode_GPU(coarse_node_index, Nx, Ny, Nz, dim) - ((Nx)*2 + 1)*3 + id%dim + (2*Nx+1)*(2*Ny+1)*3;
			counter++;
		}

		// south-east
		if ( id < (Nx+1)*(Ny+1)*(Nz)*dim && (id) % ((Nx + 1)*(Ny + 1)*dim) >= (Nx + 1)*dim && (base_id) % ((Nx*dim) + (base_id/(3*(Nx+1)))*dim*(Nx+1)) != 0 )
		{	
			r_index[counter + id*r_max_row_size] = dim*getFineNode_GPU(coarse_node_index, Nx, Ny, Nz, dim) - ((Nx)*2 + 1)*3 + dim + id%dim + (2*Nx+1)*(2*Ny+1)*3;
			counter++;
		}

		// west
		if ( id < (Nx+1)*(Ny+1)*(Nz)*dim && (id) % ((Nx + 1)*dim) >= dim )
		{
			r_index[counter + id*r_max_row_size] = dim*getFineNode_GPU(coarse_node_index, Nx, Ny, Nz, dim) - dim + id%dim + (2*Nx+1)*(2*Ny+1)*3;
			counter++;
		}

		// origin
		if ( id < (Nx+1)*(Ny+1)*(Nz)*dim )
		{
			r_index[counter + id*r_max_row_size] = dim*getFineNode_GPU(coarse_node_index, Nx, Ny, Nz, dim) + id%dim + (2*Nx+1)*(2*Ny+1)*3;
			counter++;
		}

		// CHECK:
		// east 
		if ( id < (Nx+1)*(Ny+1)*(Nz)*dim && (base_id) % ((Nx*dim) + (base_id/(3*(Nx+1)))*dim*(Nx+1)) != 0 || base_id == 0 )
		{
			r_index[counter + id*r_max_row_size] = dim*getFineNode_GPU(coarse_node_index, Nx, Ny, Nz, dim) + dim + id%dim + (2*Nx+1)*(2*Ny+1)*3;
			counter++;
		}

		// north-west
		if ( id < (Nx+1)*(Ny+1)*(Nz)*dim && (id) % ((Nx + 1)*(Ny + 1)*dim) < (Nx+1)*(Ny)*dim && (id) % ((Nx + 1)*dim) >= dim )
		{
			r_index[counter + id*r_max_row_size] = dim*getFineNode_GPU(coarse_node_index, Nx, Ny, Nz, dim) + ((Nx)*2 + 1)*3 - dim + id%dim + (2*Nx+1)*(2*Ny+1)*3;
			counter++;
		}

		// north
		if ( id < (Nx+1)*(Ny+1)*(Nz)*dim && (id) % ((Nx + 1)*(Ny + 1)*dim) < (Nx+1)*(Ny)*dim )
		{
			r_index[counter + id*r_max_row_size] = dim*getFineNode_GPU(coarse_node_index, Nx, Ny, Nz, dim) + ((Nx)*2 + 1)*3 + id%dim + (2*Nx+1)*(2*Ny+1)*3;
			counter++;
		}

		// north-east
		if ( id < (Nx+1)*(Ny+1)*(Nz)*dim && (id) % ((Nx + 1)*(Ny + 1)*dim) < (Nx+1)*(Ny)*dim && (base_id) % ((Nx*dim) + (base_id/(3*(Nx+1)))*dim*(Nx+1)) != 0 || base_id == 0 )
		{
			r_index[counter + id*r_max_row_size] = dim*getFineNode_GPU(coarse_node_index, Nx, Ny, Nz, dim) + ((Nx)*2 + 1)*3 + dim + id%dim + (2*Nx+1)*(2*Ny+1)*3;
			counter++;
		}










	for ( int i = counter ; i < r_max_row_size; i++)
	{
		r_index[i + id*r_max_row_size] = num_cols;
	}




	}


	// if ( id == 0 )
	// {
	// 	for ( int i = 0 ; i < r_max_row_size ; i++)
	// 		printf("%lu ", r_index[i + id*r_max_row_size]);

	// 	printf("\n");
	// }




}


// DEBUG:
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



// // TODO: to delete
__global__
void checkLaplacian(double* laplacian, double* chi, size_t Nx, size_t Ny, size_t Nz, size_t numElements, double h)
{
	// laplacian_GPU( double *array, size_t ind, size_t Nx, size_t Ny, size_t Nz )
	unsigned int id = threadIdx.x + blockIdx.x*blockDim.x;
	if ( id < numElements)
	{
		laplacian[id] = laplacian_GPU( chi, id, Nx, Ny, Nz, h );
	}
	
		

}

__global__ void bar(double* x)
{
	
	// for ( int i = 0 ; i < 24 ; i++)
	int i = 0;
		printf("%.5f\n", laplacian_GPU( x, i, 6, 2, 2, 0.5) );
		// laplacian_GPU( x, i, 6, 2, 2, 0.5);
}


