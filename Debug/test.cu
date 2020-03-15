#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <cuda.h>
#include <cuda_runtime.h>

using namespace std;

// #if __CUDA_ARCH__ < 600
// __device__ double atomicAdd(double* address, double val)
// {
//     unsigned long long int* address_as_ull =
//                               (unsigned long long int*)address;
//     unsigned long long int old = *address_as_ull, assumed;

//     do {
//         assumed = old;
//         old = atomicCAS(address_as_ull, assumed,
//                         __double_as_longlong(val +
//                                __longlong_as_double(assumed)));

//     // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
//     } while (assumed != old);

//     return __longlong_as_double(old);
// }
// #endif

#define CUDA_CALL( call )                                                                                          \
    {                                                                                                                  \
    cudaError_t err = call;                                                                                          \
    if ( cudaSuccess != err)                                                                                         \
        fprintf(stderr, "CUDA error for %s in %d of %s : %s.\n", #call , __LINE__ , __FILE__ ,cudaGetErrorString(err));\
    }


// __global__ void print_GPU(int* x){
    
//     printf("[GPU] (int) = %d\n", *x);
// }


__global__ void vectorAdd(double* a, double* b, size_t N)
{
    int id = blockDim.x * blockIdx.x + threadIdx.x;

    if ( id < N )
    {
        atomicAdd(a, b[id]);

    }

}

int main()
{
    size_t N = 10;
    double a = 0;
    vector<double> b(N,1);

    double* d_a;
    double* d_b;

    CUDA_CALL( cudaMalloc( (void**)&d_a, sizeof(double) ) );
    CUDA_CALL( cudaMalloc( (void**)&d_b, sizeof(double) * N ) );
    CUDA_CALL( cudaMemcpy( d_a, &a, sizeof(double), cudaMemcpyHostToDevice )    );
    CUDA_CALL( cudaMemcpy( d_b, &b[0], sizeof(double) * N, cudaMemcpyHostToDevice )    );
	
	vectorAdd<<<1,10>>>(d_a, d_b, N);

    CUDA_CALL( cudaMemcpy( &a, d_a, sizeof(double), cudaMemcpyDeviceToHost )    );

    cout << a << endl;
    
    
    cudaFree( d_a );
    cudaFree( d_b );
    
	
	cudaDeviceSynchronize();


}
