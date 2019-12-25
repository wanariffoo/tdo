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
void calcLambdaTrial(double *rho_trial, double rho, double *lambda_l, double *lambda_u, double *lambda_trial)
{
    if ( *rho_trial > rho )
        *lambda_l = *lambda_trial;

    else
        *lambda_u = *lambda_trial;

    *lambda_trial = 0.5 * ( *lambda_l + *lambda_u );
}

__global__
void checkKaiConvergence(bool *foo, double *rho_trial, double rho)
{
    if ( *rho_trial - rho < 1e-7 )
        *foo = false;

}


__device__
double laplacian_GPU( double *array, size_t ind, size_t N )
{
    double value = 4.0 * array[ind];

    // east element
    if ( (ind + 1) % N != 0 )
        value += -1.0 * array[ind + 1];
    
    // north element
    if ( ind + N < N*N )    // TODO: N*N --> dim
        value += -1.0 * array[ind + N];

    // west element
    if ( ind % N != 0 )
        value += -1.0 * array[ind - 1];

    // south element
    if ( ind >= N )
        value += -1.0 * array[ind - N];

    return value;


}




// __global__
// void calcLambdaUpper(double* lambda_u, double *p, double beta, double *laplacian, double eta)
// {


//     getMax(float *array, float *max, int *mutex, unsigned int n)

// }


__global__ 
void calcLambdaLower(double *df_array, double *min, int *mutex, double beta, double *kai, double eta, unsigned int N, unsigned int numElements)
{
	unsigned int index = threadIdx.x + blockIdx.x*blockDim.x;
	unsigned int stride = gridDim.x*blockDim.x;
	unsigned int offset = 0;

	__shared__ double cache[256];

    *min = 1.0e9;
    double temp = 1.0e9;
    

	while(index + offset < numElements){
        temp = fminf(temp, ( df_array[index + offset] + ( beta * laplacian_GPU( kai, index, N ) ) - eta ) );
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
void calcLambdaUpper(double *df_array, double *max, int *mutex, double beta, double *kai, double eta, unsigned int N, unsigned int numElements)
{
	unsigned int index = threadIdx.x + blockIdx.x*blockDim.x;
	unsigned int stride = gridDim.x*blockDim.x;
	unsigned int offset = 0;

	__shared__ double cache[256];

    *max = -1.0e9;
    double temp = -1.0e9;
    
	while(index + offset < numElements){
        // temp = fmaxf(temp, ( df_array[index + offset] + ( beta * laplacian[index] ) + eta ) );
        temp = fmaxf(temp, ( df_array[index + offset] + ( beta * laplacian_GPU( kai, index, N ) ) + eta ) );
         
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

double laplacian(double *array, size_t ind, size_t N)
{
    double value = 4.0 * array[ind];

    // east element
    if ( (ind + 1) % N != 0 )
        value += -1.0 * array[ind + 1];
    
    // north element
    if ( ind + N < N*N )    // TODO: N*N --> dim
        value += -1.0 * array[ind + N];

    // west element
    if ( ind % N != 0 )
        value += -1.0 * array[ind - 1];

    // south element
    if ( ind >= N )
        value += -1.0 * array[ind - N];

    return value;
}

// TODO: change kai to something else
__global__
void calcKaiTrial(   
    double *kai, 
    double *df, 
    double *lambda_trial, 
    double del_t,
    double eta,
    double beta,
    double* kai_trial,
    size_t N,
    size_t numElements
)
{
    unsigned int id = threadIdx.x + blockIdx.x*blockDim.x;
    
    __shared__ double del_kai[256];

    // if ( id == 0 )
    // printf("%f\n", *lambda_trial);

    if ( id < numElements )
    {
        del_kai[id] = ( del_t / eta ) * ( df[id] - *lambda_trial + beta*( laplacian_GPU( kai, id, N ) ) );
        

        if ( del_kai[id] + kai[id] > 1 )
        kai_trial[id] = 1;
        
        else if ( del_kai[id] + kai[id] < 1e-9 )
        kai_trial[id] = 1e-9;
        
        else
        kai_trial[id] = del_kai[id] + kai[id];
        
        // printf("%d %f \n", id, kai_trial[id]);
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


// __global__
// void calcRhoTrial(
//     double* rho_tr, 
//     double* lambda_l, 
//     double* lambda_u, 
//     double* lambda_tr, 
//     double rho, 
//     double volume
// {
//     int id = blockDim.x * blockIdx.x + threadIdx.x;
	
//     if(id == 0)
//         *rho_tr /= volume;
//     }
    
__global__
void calcLambdaTrial(double* lambda_tr, double* lambda_l, double* lambda_u, double* rho_tr, double rho, double volume)
{
    *rho_tr /= volume;
    // printf("%f\n", *rho_tr);

    if ( *rho_tr > rho )
    {
        *lambda_l = *lambda_tr;
        // printf("aps\n");
    }
    
    else
        *lambda_u = *lambda_tr;

    *lambda_tr = 0.5 * ( *lambda_u + *lambda_l );

}

// x[] = u[]^T * A * u[]
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
        
        x[id] *= u[ ( node_index [ id / 2 ] * 2 ) + ( id % 2 ) ];
    }
}

// df = ( 1/2*omega ) * p * kai^(p-1) * sum(local stiffness matrices)
__global__
void UpdateDrivingForce(double *df, double *uTKu, double p, double *kai, double local_volume, size_t N)
{
    int id = blockDim.x * blockIdx.x + threadIdx.x;

    if ( id < N )
        df[id] = uTKu[id] * ( 1 / (2*local_volume) ) * p * pow(kai[id], p - 1);
}

// __global__
// void UpdateDrivingForce(double *df, double p, double *kai)
// {
//         *df *= (0.5) * p * pow(*kai, p - 1);
// }


__global__
void checkRhoTrial(bool* inner_foo, double *rho_tr, double rho)
{
    if ( abs( *rho_tr - rho ) < 1e-7 )
        *inner_foo = false;

}
// calculate the driving force per element
__host__
void calcDrivingForce(
    double *df,             // driving force
    double *kai,            // design variable
    double p,               // penalization parameter
    double *temp,           // dummy/temp vector
    double *u,              // elemental displacement vector
    size_t* node_index,
    double* value,          // local ELLPack stiffness matrix's value vector
    size_t* index,          // local ELLPack stiffness matrix's index vector
    size_t max_row_size,    // local ELLPack stiffness matrix's maximum row size
    size_t num_rows,        // local ELLPack stiffness matrix's number of rows
    dim3 gridDim,           // grid and 
    dim3 blockDim)          // block sizes needed for running CUDA kernels
{

    // temp[] = u[]^T * A * u[]
    uTAu_GPU<<<gridDim, blockDim>>>(temp, u, node_index, value, index, max_row_size, num_rows);
    cudaDeviceSynchronize();

    // printVector_GPU<<<1, num_rows>>>( temp, num_rows );
    // printVector_GPU<<<1, num_rows * max_row_size>>>( value, num_rows * max_row_size );
    sumOfVector_GPU<<<gridDim, blockDim>>>(df, temp, num_rows);
    
    // UpdateDrivingForce<<<1,1>>>(df, p, kai);
    cudaDeviceSynchronize();
}

__global__
void applyMatrixBC(double *vValue, size_t *vIndex, size_t index, size_t num_rows, size_t num_cols, size_t max_row_size)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;

    if ( idy < num_rows && idx < num_cols && idx == index || idy == index )
    {
        if ( idx == idy )
        setAt( idx, idy, vValue, vIndex, max_row_size, 1.0);

        else
        setAt( idx, idy, vValue, vIndex, max_row_size, 0.0);
    }

}

void applyMatrixBC_(double *array, size_t index, size_t num_rows, size_t num_cols)
{
    for ( int i = 0 ; i < num_rows ; i++ )
    {
        for ( int j = 0 ; j < num_cols ; j++ )
        {
            if ( i == index && j == index )
                array[ i + num_cols*j ] = 1.0;
                
            else if ( i == index || j == index )
                array[ i + num_cols*j ] = 0.0;
        }
    }
}

int main()
{
    size_t num_rows = 18;
    size_t num_cols = 18;

    size_t max_row_size = 8;
    size_t N = 2;

    vector<size_t> bc_index = {0, 1, 6, 7, 12, 13};

    vector<double> globalmatrix = {
        6652102.4,	2400134.4,	-4066334.72,	-185606.4,	0,	0,	740236.8,	185606.4,	-3325952,	-2400153.6,	0,	0,	0,	0,	0,	0,	0,	0,
        2400134.4,	6652102.4,	185606.4,	740236.8,	0,	0,	-185606.4,	-4066334.72,	-2400153.6,	-3325952,	0,	0,	0,	0,	0,	0,	0,	0,
        -4066334.72,	185606.4,	13304204.8,	0,	-4066334.72,	-185606.4,	-3325952,	2400153.6,	1480473.6,	0,	-3325952,	-2400153.6,	0,	0,	0,	0,	0,	0,
        -185606.4,	740236.8,	0,	13304204.8,	185606.4,	740236.8,	2400153.6,	-3325952,	0,	-8132669.44,	-2400153.6,	-3325952,	0,	0,	0,	0,	0,	0,
        0,	0,	-4066334.72,	185606.4,	6652102.4,	-2400134.4,	0,	0,	-3325952,	2400153.6,	740236.8,	-185606.4,	0,	0,	0,	0,	0,	0,
        0,	0,	-185606.4,	740236.8,	-2400134.4,	6652102.4,	0,	0,	2400153.6,	-3325952,	185606.4,	-4066334.72,	0,	0,	0,	0,	0,	0,
        740236.8,	-185606.4,	-3325952,	2400153.6,	0,	0,	13304204.8,	0,	-8132669.44,	0,	0,	0,	740236.8,	185606.4,	-3325952,	-2400153.6,	0,	0,
        185606.4,	-4066334.72,	2400153.6,	-3325952,	0,	0,	0,	13304204.8,	0,	1480473.6,	0,	0,	-185606.4,	-4066334.72,	-2400153.6,	-3325952,	0,	0,
        -3325952,	-2400153.6,	1480473.6,	0,	-3325952,	2400153.6,	-8132669.44,	0,	26608409.6,	0,	-8132669.44,	0,	-3325952,	2400153.6,	1480473.6,	0,	-3325952,	-2400153.6,
        -2400153.6,	-3325952,	0,	-8132669.44,	2400153.6,	-3325952,	0,	1480473.6,	0,	26608409.6,	0,	1480473.6,	2400153.6,	-3325952,	0,	-8132669.44,	-2400153.6,	-3325952,
        0,	0,	-3325952,	-2400153.6,	740236.8,	185606.4,	0,	0,	-8132669.44,	0,	13304204.8,	0,	0,	0,	-3325952,	2400153.6,	740236.8,	-185606.4,
        0,	0,	-2400153.6,	-3325952,	-185606.4,	-4066334.72,	0,	0,	0,	1480473.6,	0,	13304204.8,	0,	0,	2400153.6,	-3325952,	185606.4,	-4066334.72,
        0,	0,	0,	0,	0,	0,	740236.8,	-185606.4,	-3325952,	2400153.6,	0,	0,	6652102.4,	-2400134.4,	-4066334.72,	185606.4,	0,	0,
        0,	0,	0,	0,	0,	0,	185606.4,	-4066334.72,	2400153.6,	-3325952,	0,	0,	-2400134.4,	6652102.4,	-185606.4,	740236.8,	0,	0,
        0,	0,	0,	0,	0,	0,	-3325952,	-2400153.6,	1480473.6,	0,	-3325952,	2400153.6,	-4066334.72,	-185606.4,	13304204.8,	0,	-4066334.72,	185606.4,
        0,	0,	0,	0,	0,	0,	-2400153.6,	-3325952,	0,	-8132669.44,	2400153.6,	-3325952,	185606.4,	740236.8,	0,	13304204.8,	-185606.4,	740236.8,
        0,	0,	0,	0,	0,	0,	0,	0,	-3325952,	-2400153.6,	740236.8,	185606.4,	0,	0,	-4066334.72,	-185606.4,	6652102.4,	2400134.4,
        0,	0,	0,	0,	0,	0,	0,	0,	-2400153.6,	-3325952,	-185606.4,	-4066334.72,	0,	0,	185606.4,	740236.8,	2400134.4,	6652102.4
    };

    for ( int i = 0 ; i < bc_index.size() ; ++i )
        applyMatrixBC_(&globalmatrix[0], bc_index[i], num_rows, num_cols);

    for ( int i = 0 ; i < num_rows ; i++ )
    {
        for ( int j = 0 ; j < num_cols ; j++ )
        {
            cout << globalmatrix[i + num_cols*j] << " ";
        }

        cout << "\n";
    }
    

    cudaDeviceSynchronize();
}