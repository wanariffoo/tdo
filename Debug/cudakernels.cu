
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

// sets the value of an ELLPack matrix A at (x,y)
__device__
void setAt( size_t x, size_t y, double* vValue, size_t* vIndex, size_t max_row_size, double value )
{
    for(size_t k = 0; k < max_row_size; ++k)
    {
        if(vIndex[x * max_row_size + k] == y)
        {
            vValue[x * max_row_size + k] += value;
            // printf("%f \n", vValue[x * max_row_size + k]);
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


std::size_t getMaxRowSize(std::vector<double> &array, std::size_t num_rows, std::size_t num_cols)
{
	std::size_t max_row_size = 0;

	for ( int i = 0; i < num_rows ; i++ )
	{
		std::size_t max_in_row = 0;

		for ( int j = 0 ; j < num_cols ; j++ )
		{
			if ( array[ j + i*num_cols ] != 0 )
				max_in_row++;

		}

		if ( max_in_row >= max_row_size )
			max_row_size = max_in_row;
	}

	
	return max_row_size;

}

// transforms a flattened matrix (array) to ELLPACK's vectors value and index
// max_row_size has to be d prior to this
// void transformToELL(std::vector<double> &array, std::vector<double> &value, std::vector<std::size_t> &index, size_t max_row_size, size_t num_rows)
void transformToELL(std::vector<double> &array, double* value, size_t* index, size_t max_row_size, size_t num_rows)
{
	value[0] = 20;
	// for ( int id = 0 ; id < num_rows ; id++)
    // {
    //     size_t counter = id*max_row_size;
    //     size_t nnz = 0;
        
	// 		// printf("array = %e\n", array [ 1 ]);
    //     for ( int j = 0 ; nnz < max_row_size ; j++ )
    //     {

    //         if ( array [ j + id*num_rows ] != 0 )
    //         {
	// 			// printf("array = %e\n", array [ j + id*num_rows ]);
	// 			value [counter] = array [ j + id*num_rows ];
				
					
    //             index [counter] = j;
	// 			// printf("value = %e\n", value[counter]);
    //             counter++;
    //             nnz++;
    //         }
            
    //         if ( j == num_rows - 1 )
    //         {
    //             for ( ; nnz < max_row_size ; counter++ && nnz++ )
    //             {
    //                 value [counter] = 0.0;
    //                 index [counter] = num_rows;
    //             }
    //         }
    //     }
	// }
	
}

// sets identity rows and columns of the DOF in which a BC is applied
void applyMatrixBC(vector<vector<double>> &array, size_t index, size_t num_rows)
{
    for ( int i = 0 ; i < num_rows ; i++ )
    {
        for ( int j = 0 ; j < num_rows ; j++ )
        {
            if ( i == index && j == index )
                array[i][j] = 1.0;
                
            else if ( i == index || j == index )
                array[i][j] = 0.0;
        }
    }
}


// // __global__ void getMax(double *array, double *max, int *mutex, unsigned int n)
// // {
// // 	unsigned int index = threadIdx.x + blockIdx.x*blockDim.x;
// // 	unsigned int stride = gridDim.x*blockDim.x;
// // 	unsigned int offset = 0;

// // 	__shared__ double cache[blockDim.x];


// // 	double temp = -1.0;
// // 	while(index + offset < n){
// // 		temp = fmaxf(temp, array[index + offset]);

// // 		offset += stride;
// // 	}

// // 	cache[threadIdx.x] = temp;

// // 	__syncthreads();


// // 	// reduction
// // 	unsigned int i = blockDim.x/2;
// // 	while(i != 0){
// // 		if(threadIdx.x < i){
// // 			cache[threadIdx.x] = fmaxf(cache[threadIdx.x], cache[threadIdx.x + i]);
// // 		}

// // 		__syncthreads();
// // 		i /= 2;
// // 	}

// // 	if(threadIdx.x == 0){
// // 		while(atomicCAS(mutex,0,1) != 0);  //lock
// // 		*max = fmaxf(*max, cache[0]);
// // 		atomicExch(mutex, 0);  //unlock
// // 	}
// // }


// a = b
__global__ 
void vectorEquals_GPU(double* a, double* b, size_t num_rows)
{
	int id = blockDim.x * blockIdx.x + threadIdx.x;

	if ( id < num_rows )
		a[id] = b[id];
}





// ////////////////////////////////////////////
// // SMOOTHERS
// ////////////////////////////////////////////

__global__ void Jacobi_Precond_GPU(double* c, double* value, size_t* index, size_t max_row_size, double* r, size_t num_rows){

	int id = blockDim.x * blockIdx.x + threadIdx.x;

	if ( id < num_rows )
		c[id] = valueAt(id, id, value, index, max_row_size) * r[id];
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