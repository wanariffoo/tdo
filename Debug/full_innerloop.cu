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
void calcInnerLoop(double* n, double h, double* eta, double* beta)
{
    *n = ( 6 / *eta ) * ( *beta / ( h * h ) );
}


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
		atomicAdd_double(sum, cache[0]);
}

// x[] = u[]^T * A * u[]
__global__
void uTAu_GPU(double *x, double *u, double* value, size_t* index, size_t max_row_size, size_t num_rows)
{
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    
    if ( id < num_rows )
    {
        x[id] = 0;
        for ( int n = 0; n < max_row_size; n++ )
		{
            int col = index [ max_row_size * id + n ];
			double val = value [ max_row_size * id + n ];
            x[id] += val * u [ col ];
        }
        
        x[id] *= u[id];
        
    }
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
    // cudaDeviceSynchronize();

    // printVector_GPU<<<1, num_rows>>>( u, num_rows );
    // printVector_GPU<<<1, num_rows * max_row_size>>>( value, num_rows * max_row_size );
    // cudaDeviceSynchronize();
    
    sumOfVector_GPU<<<gridDim, blockDim>>>(df, temp, num_rows);
    cudaDeviceSynchronize();

    UpdateDrivingForce<<<1,1>>>(df, p, kai);
}


int main()
{
    size_t num_rows = 8;
    size_t num_GP = 4;
    size_t max_row_size = 8;

    // rho
    double rho = 0.4;

    // displacement vector
    vector<double> u = { 0, 0, -0.044797, -0.026485, 0, 0, -0.044798, -0.026485};
    vector<double> temp(num_rows, 0.0);
    
    // double *d_u;

    // CUDA_CALL ( cudaMalloc( (void**)&d_u, sizeof(double) * num_rows) );
    // CUDA_CALL ( cudaMemcpy( d_u, &u[0], sizeof(double) * num_rows, cudaMemcpyHostToDevice) );


    // inner loop
    double eta = 12;
    double beta = 1;
    double h = 0.5;

    // driving force
    double kai = 0.4;
    double df;
    vector<double> p(0, num_GP);
    
    // bisection
    double del_t = 1;


    vector<double> l_value = {
        74875200,	23550100,	-40318880,	6069950,	-58414000,	-40627200,	23857900,	11006650,
        12680850,	74875200,	21876350,	23857900,	-21876350,	-58414000,	-12680850,	-40318880,
        -40318880,	11006650,	86534200,	-40626900,	12199100,	-5588950,	-58414000,	35209400,
        -12680850,	23857900,	-21876350,	108186200,	21876350,	-73629940,	12680850,	-58414000,
        -58414000,	-40627200,	12199100,	1013350,	119845000,	45202400,	-73629940,	-5588950,
        -21876350,	-58414000,	-24340100,	-73629940,	24340100,	119845000,	21876350,	12199100,
        23857900,	6069950,	-58414000,	33543400,	-73629940,	1013350,	108186200,	-40626900,
        21876350,	-40318880,	24340100,	-58414000,	-24340100,	12199100,	-21876350,	86534200
    };

    vector<size_t> l_index = {
        0,	1,	2,	3,	4,	5,	6,	7,
        0,	1,	2,	3,	4,	5,	6,	7,
        0,	1,	2,	3,	4,	5,	6,	7,
        0,	1,	2,	3,	4,	5,	6,	7,
        0,	1,	2,	3,	4,	5,	6,	7,
        0,	1,	2,	3,	4,	5,	6,	7,
        0,	1,	2,	3,	4,	5,	6,	7,
        0,	1,	2,	3,	4,	5,	6,	7
    };


    // CUDA

    double *d_eta;
    double *d_n;
    double *d_beta;
    double *d_kai;
    double *d_df;

    // double *d_p;
    double *d_temp;
    double *d_u;

    double *d_l_value;
    size_t *d_l_index;

    CUDA_CALL ( cudaMalloc( (void**)&d_eta, sizeof(double) ) );
    CUDA_CALL ( cudaMalloc( (void**)&d_n, sizeof(double) ) );
    CUDA_CALL ( cudaMalloc( (void**)&d_beta, sizeof(double) ) );
    CUDA_CALL ( cudaMalloc( (void**)&d_df, sizeof(double) ) );
    CUDA_CALL ( cudaMalloc( (void**)&d_kai, sizeof(double) ) );
    CUDA_CALL ( cudaMalloc( (void**)&d_temp, sizeof(double) * num_rows) );
    CUDA_CALL ( cudaMalloc( (void**)&d_u, sizeof(double) * num_rows) );
    CUDA_CALL ( cudaMalloc( (void**)&d_l_value, sizeof(double) * num_rows * max_row_size ) );
    CUDA_CALL ( cudaMalloc( (void**)&d_l_index, sizeof(size_t) * num_rows * max_row_size ) );
    
    CUDA_CALL ( cudaMemset( d_n, 0, sizeof(double) ) );
    CUDA_CALL ( cudaMemset( d_df, 0, sizeof(double) ) );
    CUDA_CALL ( cudaMemcpy( d_kai, &kai, sizeof(double), cudaMemcpyHostToDevice) );
    CUDA_CALL ( cudaMemcpy( d_eta, &eta, sizeof(double), cudaMemcpyHostToDevice) );
    CUDA_CALL ( cudaMemcpy( d_beta, &beta, sizeof(double), cudaMemcpyHostToDevice) );
    CUDA_CALL ( cudaMemcpy( d_u, &u[0], sizeof(double) * num_rows, cudaMemcpyHostToDevice) );
    CUDA_CALL ( cudaMemcpy( d_temp, &temp[0], sizeof(double) * num_rows, cudaMemcpyHostToDevice) );

    CUDA_CALL ( cudaMemcpy( d_l_value, &l_value[0], sizeof(double) * num_rows * max_row_size, cudaMemcpyHostToDevice) );
    CUDA_CALL ( cudaMemcpy( d_l_index, &l_index[0], sizeof(size_t) * num_rows * max_row_size, cudaMemcpyHostToDevice) );

    // get block and grid dimensions
    dim3 gridDim;
    dim3 blockDim;
    calculateDimensions( num_rows, gridDim, blockDim );
    


    ///////////////////////////////////////////////////////////////////////////////////////
    // start inner loop when you have u vector
    ///////////////////////////////////////////////////////////////////////////////////////

    calcInnerLoop<<<1,1>>>( d_n, h, d_eta, d_beta );
    
    // printVector_GPU<<<1,num_rows>>>(d_u, num_rows);
    calcDrivingForce ( d_df, d_kai, 3, d_temp, d_u, d_l_value, d_l_index, max_row_size, num_rows, gridDim, blockDim );
    

    

    
    
    print_GPU<<<1,1>>>(d_df);
    cudaDeviceSynchronize();
    
}