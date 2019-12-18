/*
    
*/

#include <iostream>
#include <cuda.h>
#include <vector>
#include <cuda_runtime.h>
#include "../include/mycudaheader.h"

using namespace std;

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
void calcLambdaLower(double *array, double *min, int *mutex, double beta, double *laplacian, double eta, unsigned int n)
{
	unsigned int index = threadIdx.x + blockIdx.x*blockDim.x;
	unsigned int stride = gridDim.x*blockDim.x;
	unsigned int offset = 0;

	__shared__ double cache[256];

    *min = 1.0e9;
    double temp = 1.0e9;
    

	while(index + offset < n){
        temp = fminf(temp, ( array[index + offset] + ( beta * laplacian[index] ) - eta ) );
        
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
void calcLambdaUpper(double *array, double *max, int *mutex, double beta, double *laplacian, double eta, unsigned int n)
{
	unsigned int index = threadIdx.x + blockIdx.x*blockDim.x;
	unsigned int stride = gridDim.x*blockDim.x;
	unsigned int offset = 0;

	__shared__ double cache[256];

    *max = -1.0e9;
    double temp = -1.0e9;
    

	while(index + offset < n){
        temp = fmaxf(temp, ( array[index + offset] + ( beta * laplacian[index] ) + eta ) );
        
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
void getKaiTrial(   
    double *kai, 
    double *p, 
    double *lambda_trial, 
    double del_t,
    double eta,
    double beta,
    double* kai_trial,
    size_t numElements
)
{
    unsigned int id = threadIdx.x + blockIdx.x*blockDim.x;
    
    __shared__ double del_kai[256];

    del_kai[id] = ( del_t / eta ) * ( p[id] - *lambda_trial + beta*( laplacian_GPU( kai, id, numElements ) ) );

    if ( del_kai[id] + kai[id] > 1 )
        kai_trial[id] = 1;

    else if ( del_kai[id] + kai[id] < 1e-9 )
        kai_trial[id] = 1e-9;

    else
        kai_trial[id] = del_kai[id] + kai[id];
}

__global__ 
void sumOfArray_GPU(double* sum, double* x, size_t n)
{
    int id = threadIdx.x + blockDim.x*blockIdx.x;
	int stride = blockDim.x*gridDim.x;
   
    __shared__ double cache[1024];
    cache[id] = 0;
    
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

int main()
{
    size_t N = 2;
    double rho = 0.4;
    double lambda_trial = 0;
    double lambda_min;
    double lambda_max;
    double del_t = 1;
    double eta = 12;
    double beta = 0.5;
    size_t numElements = 4;
    
    vector<double> elements = { 0.4, 0.4, 0.4, 0.4 };
    vector<double> laplace_array(4);
    double* h_elements = &elements[0];

    vector<double> p = { 1.0, 2.03, 3.09, 9.05 };

    double* h_p = &p[0];

    // for ( int i = 0 ; i < numElements ; i++ )
    // {
    //     laplace_array[i] = laplacian( h_elements, i, N );
    //     cout << laplace_array[i] << endl;
    // }

    // CUDA

    double *d_elements;
    double *d_kai_trial;
    double *d_laplacian;
    double *d_dummy;
    double *d_rho_trial;
    double *d_rho;
    double *d_max;
    double *d_p;
    int *d_mutex;
    double *d_lambda_l;
    double *d_lambda_u;
    double *d_lambda_tr;
    

    cudaMalloc( (void**)&d_laplacian, sizeof(double) * 4 );
    cudaMalloc( (void**)&d_elements, sizeof(double) * 4 );
    cudaMalloc( (void**)&d_kai_trial, sizeof(double) * 4 );
    cudaMalloc( (void**)&d_p, sizeof(double) * 4 );
    cudaMalloc( (void**)&d_rho_trial, sizeof(double) );
    cudaMalloc( (void**)&d_rho, sizeof(double) );
    cudaMalloc( (void**)&d_dummy, sizeof(double) * 4 );
    cudaMalloc( (void**)&d_mutex, sizeof(int) );
    cudaMalloc( (void**)&d_max, sizeof(double) );
    cudaMalloc( (void**)&d_lambda_l, sizeof(double) );
    cudaMalloc( (void**)&d_lambda_u, sizeof(double) );
    cudaMalloc( (void**)&d_lambda_tr, sizeof(double) );
    
	cudaMemset( d_mutex, 0, sizeof(int) );
    cudaMemset( d_max, 0, sizeof(double) );
    cudaMemset( d_rho, 0, sizeof(double) );
    cudaMemset( d_rho_trial, 0, sizeof(double) );
    cudaMemset( d_lambda_tr, 0, sizeof(double) );
    cudaMemset( d_lambda_u, 0, sizeof(double) );
    cudaMemset( d_lambda_l, 0, sizeof(double) );
    
    cudaMemcpy(d_elements, &elements[0], sizeof(double) * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_laplacian, &laplace_array[0], sizeof(double) * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_p, &p[0], sizeof(double) * 4, cudaMemcpyHostToDevice);
    
    
    // can run these concurrently
    calcLambdaUpper<<< 1, 4 >>>(d_p, d_lambda_u, d_mutex, 1.0, d_laplacian, 12, 4);
    calcLambdaLower<<< 1, 4 >>>(d_p, d_lambda_l, d_mutex, 1.0, d_laplacian, 12, 4);
    
    bool foo = true;
    bool *d_foo;
    cudaMalloc( (void**)&d_foo, sizeof(bool) );
    cudaMemset( d_foo, 1, sizeof(bool) );
    
    while ( foo )
    {
        getKaiTrial<<<1,4>>>(d_elements, d_p, d_lambda_tr, del_t, eta, beta, d_kai_trial, 4);
        
        // calcRhoTrial = sum
        sumOfArray_GPU<<<1,4>>>(d_rho_trial, d_elements, 4);
        
        // determine lambda_trial
        calcLambdaTrial<<< 1,1 >>> ( d_rho_trial, rho, d_lambda_l, d_lambda_u, d_lambda_tr );
        
        checkKaiConvergence<<<1,1>>>(d_foo, d_rho_trial, rho);
        cudaDeviceSynchronize();
        cudaMemcpy(&foo, d_foo, sizeof(bool), cudaMemcpyDeviceToHost);
    }
    
    print_GPU<<<1,1>>>(d_lambda_tr);
    cudaDeviceSynchronize();
    
}