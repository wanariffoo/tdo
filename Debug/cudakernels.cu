
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
__host__ void calculateDimensions2D(size_t N, dim3 &gridDim, dim3 &blockDim)
{
    if ( N <= 1024 )
    {
        blockDim.x = 32; blockDim.y = 32; blockDim.z = 1;
        gridDim.x  = 1; gridDim.y = 1; gridDim.z = 1;
    }
        
    else
    {
        blockDim.x = 32; blockDim.y = 32; blockDim.z = 1;
        gridDim.x  = (int)ceil((N/2)/blockDim.x)+1; gridDim.y = (int)ceil((N/2)/blockDim.y)+1; gridDim.z = 1;
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
				foo[i][j] += A[i][k] * P[k][j];
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
void uTAu_GPU(double *x, double *u, size_t *node_index, double* d_A_local, size_t num_rows)
{
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    
    if ( id < num_rows )
    {
        x[id] = 0;
        for ( int n = 0; n < num_rows; n++ )
		{
            int global_col = ( node_index [ n / 2 ] * 2 ) + ( n % 2 ); // converts local node to global node
            x[id] += u[global_col] * d_A_local[ id + n*num_rows ];

			// if ( id == 3 )
			// 	printf("%e\n", x[0]);
        }

        x[id] *= u[ ( node_index [ id / 2 ] * 2 ) + ( id % 2 ) ];

    }


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

// calculate the driving force per element
__host__
void calcDrivingForce(
    double *df,             // driving force
    double *chi,            // design variable
    double p,               // penalization parameter
    double *uTAu,           // dummy/temp vector
    double *u,              // elemental displacement vector
    size_t* node_index,
	double* d_A_local,
    size_t num_rows,        // local ELLPack stiffness matrix's number of rows
    dim3 gridDim,           // grid and 
    dim3 blockDim)          // block sizes needed for running CUDA kernels
{

    // temp[] = u[]^T * A * u[]
    uTAu_GPU<<<gridDim, blockDim>>>(uTAu, u, node_index, d_A_local, num_rows);
    cudaDeviceSynchronize();

	// printVector_GPU<<<1,4>>>( u, 4);


	// calculates the driving force in each element
    sumOfVector_GPU<<<gridDim, blockDim>>>(df, uTAu, num_rows);
    cudaDeviceSynchronize();
    
	//TODO: det_J not implemented yet
	// df[] *= ( 1 / 2*omega ) * ( p * pow(chi[], p - 1 ) * det(J)
    // UpdateDrivingForce<<<gridDim,blockDim>>>(df, p, chi, local_volume, num_rows);
	
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


__global__ void calcRhoTrial(double* chi_trial, double local_volume, size_t numElements)
{
	double total_volume = local_volume * numElements;

	*chi_trial *= local_volume;
	*chi_trial /= total_volume;

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
void calcP_w(double* p_w, double* df, double* uTAu, double* chi, int p, double local_volume, size_t numElements)
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

__global__ void calcEtaBeta( double* eta, double* beta, double etastar, double betastar, double* p_w )
{
	unsigned int id = threadIdx.x + blockIdx.x*blockDim.x;

	if ( id == 0 )	
		*eta = etastar * (*p_w);

	if ( id == 1 )
		*beta = betastar * (*p_w);

}

__global__ void RA(	double* p_value, size_t* p_index, size_t p_max_row_size, 
					double* value, size_t* index, size_t max_row_size,
					double* temp_matrix, size_t num_rows, size_t num_cols)
{
	unsigned int idx = threadIdx.x + blockIdx.x*blockDim.x;
	unsigned int idy = threadIdx.y + blockIdx.y*blockDim.y;

	if ( idx < num_cols && idy < num_rows )
	{	
		
		for ( int j = 0 ; j < num_cols ; ++j )
		{
			temp_matrix[idx + idy*num_cols] += valueAt(j, idy, p_value, p_index, p_max_row_size) * valueAt(idx, j, value, index, max_row_size);
			// temp_matrix[idx + idy*num_cols] += valueAt(idy, j, r_value, r_index, r_max_row_size) * valueAt(idx, j, value, index, max_row_size);

			
			// if ( idx == 8 && idy == 2)
				// printf("%f\n", temp_matrix[44]);
		}
			// temp_matrix[idx + idy*num_cols] += valueAt(j, idy, r_value, r_index, r_max_row_size) * valueAt(idx, j, value, index, max_row_size);

		// if ( idx == 8 && idy == 2)
		// 		printf("%f\n", valueAt(2, 2, r_value, r_index, r_max_row_size));

	}

}


__global__ void AP(	double* value, size_t* index, size_t max_row_size, 
					double* p_value, size_t* p_index, size_t p_max_row_size, 
					double* temp_matrix, size_t num_rows, size_t num_cols)
{
	unsigned int idx = threadIdx.x + blockIdx.x*blockDim.x;
	unsigned int idy = threadIdx.y + blockIdx.y*blockDim.y;

	if ( idx < num_cols && idy < num_rows )
	{	
		
		for ( int j = 0 ; j < num_cols ; ++j )
		{
			// temp_matrix[idx + idy*num_cols] += valueAt(idy, j, r_value, r_index, r_max_row_size) * valueAt(idx, j, value, index, max_row_size);

			// if ( idx == 3 && idy == 2 )
			// {
			// 	printf("%f\n", valueAt(j, idx, p_value, p_index, p_max_row_size));
			// }
				
				addAt( idx, idy, value, index, max_row_size, temp_matrix[j + idy*num_cols] * valueAt(j, idx, p_value, p_index, p_max_row_size) );
			

		}
			// temp_matrix[idx + idy*num_cols] += valueAt(j, idy, r_value, r_index, r_max_row_size) * valueAt(idx, j, value, index, max_row_size);

		// if ( idx == 0 && idy == 0)
		// 		printf("%f\n", valueAt(0, 0, value, index, max_row_size));

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
	// unsigned int id = threadIdx.x + blockIdx.x*blockDim.x;

	dim3 gridDim;
    dim3 blockDim;
	calculateDimensions2D( num_rows[0] * num_rows[1], gridDim, blockDim);
	
	// temp_matrix = R * A_fine
	RA<<<gridDim,blockDim>>>(p_value[0], p_index[0], p_max_row_size[0], value[1], index[1], max_row_size[1], temp_matrix, num_rows[0], num_rows[1]);
	cudaDeviceSynchronize();

	// calculateDimensions2D( num_rows[0] * num_rows[0], gridDim, blockDim);
	AP<<<gridDim,blockDim>>>( value[0], index[0], max_row_size[0], p_value[0], p_index[0], p_max_row_size[0], temp_matrix, num_rows[0], num_rows[1]);
	cudaDeviceSynchronize();

	// A_coarse = temp_matrix * P


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
