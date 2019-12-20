/*
    
*/

#include <iostream>
#include <cuda.h>
#include <vector>
#include <cuda_runtime.h>
#include "../include/mycudaheader.h"
#include <cmath>

using namespace std;


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
    
    // reset id
    id = threadIdx.x + blockDim.x*blockIdx.x;
    // if ( id == 0)
    // if ( id == 0)
        // printf("id = %d : %f\n", id, cache[threadIdx.x]);
    // printf("id = %d\n", threadIdx.x + (blockDim.x*blockIdx.x) );
	
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

// x[] = u[]^T * A * u[]
__global__
void uTAu_GPU(double *x, double *u, double* value, size_t* index, size_t max_row_size, size_t num_rows)
{
    int id = blockDim.x * blockIdx.x + threadIdx.x;

    x[id] = 0;

    if ( id < num_rows )
    {
		for ( int n = 0; n < max_row_size; n++ )
		{
			int col = index [ max_row_size * id + n ];
			double val = value [ max_row_size * id + n ];
            x[id] += val * u [ col ];
        }
    }

    x[id] *= u[id];

    // printf("id = %d : %f\n", id, x[id]);
    
}


// df = ( 1/2*omega ) * p * kai^(p-1) * sum(local stiffness matrices)
__global__
void UpdateDrivingForce(double *df, double p, double *kai)
{
    *df *= (0.5) * p * pow(*kai, p - 1);
}

// calculate the driving force per element
__host__
void calcDrivingForce(
    double *df,             // driving force
    double *kai,            // design variable
    double p,               // penalization parameter
    double *temp,           // dummy/temp vector
    double *u,              // elemental displacement vector
    double* value,          // local ELLPack stiffness matrix's value vector
    size_t* index,          // local ELLPack stiffness matrix's index vector
    size_t max_row_size,    // local ELLPack stiffness matrix's maximum row size
    size_t num_rows,        // local ELLPack stiffness matrix's number of rows
    dim3 gridDim,           // grid and 
    dim3 blockDim)          // block sizes needed for running CUDA kernels
{
    // temp[] = u[]^T * A * u[]
    uTAu_GPU<<<gridDim, blockDim>>>(temp, u, value, index, max_row_size, num_rows);
    cudaDeviceSynchronize();
    
    // df = sum( temp[] )
    sumOfVector_GPU<<<gridDim, blockDim>>>(df, temp, num_rows);
    cudaDeviceSynchronize();

    UpdateDrivingForce<<<1,1>>>(df, p, kai);
}





int main()
{
    
    // A matrix
    vector<double> value;
    vector<size_t> index;
    size_t max_row_size = 1;
    size_t num_rows = 2000000;

    // u vector
    vector<double> u;
    vector<double> temp;

    double p = 3;
    double kai = 0.8;

    for ( int i = 0 ; i < num_rows; i++ )
    {
        value.push_back(1);
        index.push_back(0);
    }    

    for ( int i = 0 ; i < num_rows ; i++ )
    {
        u.push_back(1);
        temp.push_back(0);
    }

    // CUDA
    double *d_value;
    size_t *d_index;
    double *d_u;
    double *d_df;
    double *d_temp;
    double *d_kai;
    

    cudaMalloc( (void**)&d_value, sizeof(double) * max_row_size * num_rows );
    cudaMalloc( (void**)&d_index, sizeof(size_t) * max_row_size * num_rows );
    cudaMalloc( (void**)&d_u, sizeof(double) * num_rows );
    cudaMalloc( (void**)&d_temp, sizeof(double) * num_rows );
    cudaMalloc( (void**)&d_df, sizeof(double));
    cudaMalloc( (void**)&d_kai, sizeof(double));
    
    cudaMemset( d_df, 0, sizeof(double) );
    
    cudaMemcpy(d_value, &value[0], sizeof(double) * max_row_size * num_rows, cudaMemcpyHostToDevice);
    cudaMemcpy(d_index, &index[0], sizeof(size_t) * max_row_size * num_rows, cudaMemcpyHostToDevice);
    cudaMemcpy(d_u, &u[0], sizeof(double) * num_rows, cudaMemcpyHostToDevice);
    cudaMemcpy(d_temp, &temp[0], sizeof(double) * num_rows, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kai, &kai, sizeof(double), cudaMemcpyHostToDevice);
    
    
    dim3 blockDim;
    dim3 gridDim;

    calculateDimensions(num_rows, blockDim, gridDim);


    // run kernel
    calcDrivingForce(d_df, d_kai, p, d_temp, d_u, d_value, d_index, max_row_size, num_rows, gridDim, blockDim);
    cudaDeviceSynchronize();

    // void calcDrivingForce(
    //     double *df,             // driving force
    //     double *kai,            // design variable
    //     double *p,              // penalization parameter
    //     double *temp,           // dummy/temp vector
    //     double *u,              // elemental displacement vector
    //     double* value,          // local ELLPack stiffness matrix's value vector
    //     size_t* index,          // local ELLPack stiffness matrix's index vector
    //     size_t max_row_size,    // local ELLPack stiffness matrix's maximum row size
    //     size_t num_rows,        // local ELLPack stiffness matrix's number of rows
    //     dim3 gridDim,           // grid and 
    //     dim3 blockDim)          // block sizes needed for running CUDA kernels
    // {
    
    
    print_GPU<<<1,1>>>(d_df);
    cudaDeviceSynchronize();
    
}