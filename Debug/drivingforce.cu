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
void uTAu_GPU(double *x, double *u, size_t *node_index, double* value, size_t* index, size_t max_row_size, size_t num_rows)
{
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    
    if ( id < num_rows )
    {
        
        x[id] = 0;
        
        for ( int n = 0; n < max_row_size; n++ )
		{
            
            int col = index [ max_row_size * id + n ];
            
            int global_col = ( node_index [ col / 2 ] * 2 ) + ( col % 2 ); // converts local node to global node
            
			double val = value [ max_row_size * id + n ];
            x[id] += val * u [ global_col ];
            
        }
        x[id] *= u[id];
    }


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
    size_t *node_index,
    double* value,          // local ELLPack stiffness matrix's value vector
    size_t* index,          // local ELLPack stiffness matrix's index vector
    size_t max_row_size,    // local ELLPack stiffness matrix's maximum row size
    size_t num_rows,        // local ELLPack stiffness matrix's number of rows
    dim3 gridDim,           // grid and 
    dim3 blockDim)          // block sizes needed for running CUDA kernels
{
    
    // printVector_GPU<<<1,4>>>(node_index, 4);

    // temp[] = u[]^T * A * u[]
    uTAu_GPU<<<gridDim, blockDim>>>(temp, u, node_index, value, index, max_row_size, num_rows);
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
    size_t num_rows = 8;

    // u vector
    vector<double> u;
    vector<double> temp (num_rows, 0);

    double p = 3;
    double kai = 0.4;

    for ( int i = 0 ; i < num_rows; i++ )
    {
        value.push_back(i);
        index.push_back(i);
    }    

    for ( int i = 0 ; i < 18 ; i++ )
    {
        u.push_back(i);
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
    cudaMalloc( (void**)&d_u, sizeof(double) * 18 );
    cudaMalloc( (void**)&d_temp, sizeof(double) * num_rows );
    cudaMalloc( (void**)&d_df, sizeof(double));
    cudaMalloc( (void**)&d_kai, sizeof(double));
    
    cudaMemset( d_df, 0, sizeof(double) );
    
    cudaMemcpy(d_value, &value[0], sizeof(double) * max_row_size * num_rows, cudaMemcpyHostToDevice);
    cudaMemcpy(d_index, &index[0], sizeof(size_t) * max_row_size * num_rows, cudaMemcpyHostToDevice);
    cudaMemcpy(d_u, &u[0], sizeof(double) * 18, cudaMemcpyHostToDevice);
    cudaMemcpy(d_temp, &temp[0], sizeof(double) * num_rows, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kai, &kai, sizeof(double), cudaMemcpyHostToDevice);
    
    
    dim3 blockDim;
    dim3 gridDim;

    calculateDimensions(num_rows, blockDim, gridDim);

    // node index

    vector<size_t> node_index = {0, 1, 3, 4};

    size_t* d_node_index;
    cudaMalloc( (void**)&d_node_index, sizeof(size_t) * 4 );
    cudaMemcpy(d_node_index, &node_index[0], sizeof(size_t) * 4, cudaMemcpyHostToDevice);




    // run kernel
    calcDrivingForce(d_df, d_kai, p, d_temp, d_u, d_node_index, d_value, d_index, max_row_size, num_rows, gridDim, blockDim);
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