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
void calcLambdaUpper(double *array, double *max, int *mutex, double beta, double *laplacian, double eta, unsigned int n)
{
	unsigned int index = threadIdx.x + blockIdx.x*blockDim.x;
	unsigned int stride = gridDim.x*blockDim.x;
	unsigned int offset = 0;

	__shared__ double cache[256];

    
	double temp = -1.0e5;
	while(index + offset < n){
        temp = fmaxf(temp, ( array[index + offset] + beta * laplacian[index] - eta ) );
        
		offset += stride;
	}
    
   
    
	cache[threadIdx.x] = temp;

    if ( index == 0 )
    printf("%e\n", cache[threadIdx.x]);
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
    double *d_laplacian;
    double *d_dummy;
    double *d_max;
    double *d_p;
    int *d_mutex;
    double *d_lambda_l;
    double *d_lambda_u;
    double *d_lambda_tr;

    cudaMalloc( (void**)&d_laplacian, sizeof(double) * 4 );
    cudaMalloc( (void**)&d_elements, sizeof(double) * 4 );
    cudaMalloc( (void**)&d_p, sizeof(double) * 4 );
    cudaMalloc( (void**)&d_dummy, sizeof(double) * 4 );
    cudaMalloc( (void**)&d_mutex, sizeof(int) );
    cudaMalloc( (void**)&d_max, sizeof(double) );
    cudaMalloc( (void**)&d_lambda_l, sizeof(double) );
    cudaMalloc( (void**)&d_lambda_u, sizeof(double) );
    cudaMalloc( (void**)&d_lambda_tr, sizeof(double) );
	cudaMemset( d_mutex, 0, sizeof(int) );
    cudaMemset( d_max, 0, sizeof(double) );
    cudaMemset( d_lambda_tr, 0, sizeof(double) );
    cudaMemset( d_lambda_u, 0, sizeof(double) );
    cudaMemset( d_lambda_l, 0, sizeof(double) );

    cudaMemcpy(d_elements, &elements[0], sizeof(double) * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_laplacian, &laplace_array[0], sizeof(double) * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_p, &p[0], sizeof(double) * 4, cudaMemcpyHostToDevice);
    
    calcLambdaUpper<<< 1, 4 >>>(d_p, d_max, d_mutex, 1.0, d_laplacian, 12, 4);

    print_GPU<<<1,1>>>( d_max );
    cudaDeviceSynchronize();

    

    // cout << laplacian(h_array, 0, 3) << endl;
    // cout << laplacian(h_array, 1, 3) << endl;
    // cout << laplacian(h_array, 2, 3) << endl;
    // cout << laplacian(h_array, 3, 3) << endl;
    // cout << laplacian(h_array, 4, 3) << endl;
    // cout << laplacian(h_array, 5, 3) << endl;
    // cout << laplacian(h_array, 6, 3) << endl;
    // cout << laplacian(h_array, 7, 3) << endl;
    // cout << laplacian(h_array, 8, 3) << endl;

}