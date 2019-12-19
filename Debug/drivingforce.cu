/*
    
*/

#include <iostream>
#include <cuda.h>
#include <vector>
#include <cuda_runtime.h>
#include "../include/mycudaheader.h"

using namespace std;



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

__global__ 
void sumOfVector_GPU(double* sum, double* x, size_t n)
{
	int id = threadIdx.x + blockDim.x*blockIdx.x;
	int stride = blockDim.x*gridDim.x;
		
	__shared__ double cache[1024];
	
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

	// reset id
	id = threadIdx.x + blockDim.x*blockIdx.x;

	// reduce sum from all blocks' cache
	if(threadIdx.x == 0)
		atomicAdd_double(sum, cache[0]);
}

// x[] = u[]^T * A * u[]
__global__
void uTAu_GPU(double *x, double *u, double* value, size_t* index, size_t max_row_size, size_t num_rows)
{
    int id = blockDim.x * blockIdx.x + threadIdx.x;

    __shared__ double cache[1024];
    
    cache[threadIdx.x] = 0;

    if ( id < num_rows )
    {
		for ( int n = 0; n < max_row_size; n++ )
		{
			int col = index [ max_row_size * id + n ];
			double val = value [ max_row_size * id + n ];
            cache[threadIdx.x] += val * u [ col ];
            
        }
        
    }
    cache[threadIdx.x] *= u[id];
        // printf("%d : %e\n", id, temp[threadIdx.x]);
        // printf("%d : %e\n", id, x[id]);

        //  reduce shared variables in each block

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
    atomicAdd_double(x, cache[0]);
   

}






int main()
{
    
    // A matrix
    vector<double> value = {1,2,2,5,3,0};
    vector<size_t> index = {0,1,0,1,2,3};
    size_t max_row_size = 2;
    size_t num_rows = 3;

    // u vector
    vector<double> u = { 1, 2, 3 };

    // CUDA
    double *d_value;
    size_t *d_index;
    double *d_u;
    double *d_p;
    

    cudaMalloc( (void**)&d_value, sizeof(double) * max_row_size * num_rows );
    cudaMalloc( (void**)&d_index, sizeof(size_t) * max_row_size * num_rows );
    cudaMalloc( (void**)&d_u, sizeof(double) * num_rows );
    cudaMalloc( (void**)&d_p, sizeof(double));
    
    cudaMemset( d_p, 0, sizeof(double) );
    
    cudaMemcpy(d_value, &value[0], sizeof(double) * max_row_size * num_rows, cudaMemcpyHostToDevice);
    cudaMemcpy(d_index, &index[0], sizeof(size_t) * max_row_size * num_rows, cudaMemcpyHostToDevice);
    cudaMemcpy(d_u, &u[0], sizeof(double) * num_rows, cudaMemcpyHostToDevice);
    
    
    // run kernel
    uTAu_GPU<<<1, 1024>>>(d_p, d_u, d_value, d_index, max_row_size, num_rows);
    cudaDeviceSynchronize();
    
    
    print_GPU<<<1,1>>>(d_p);
    cudaDeviceSynchronize();
    
}