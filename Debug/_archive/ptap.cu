// #include "../cudakernels.h"
#include <iostream>
#include <vector>
#include <cmath>

using namespace std;

#define CUDA_CALL( call )                                                                                          \
    {                                                                                                                  \
    cudaError_t err = call;                                                                                          \
    if ( cudaSuccess != err)                                                                                         \
        fprintf(stderr, "CUDA error for %s in %d of %s : %s.\n", #call , __LINE__ , __FILE__ ,cudaGetErrorString(err));\
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

__global__
void PTAP_GPU(  double* value, size_t* index, size_t mrs,       // A matrix
                double* p_value, size_t* p_index, size_t p_mrs,  // P matrix
                double* a_value, size_t* a_index, size_t a_mrs  // A_ matrix (coarse)
                )
{

    printf("%e\n", valueAt( 3, 3, p_value, p_index, p_mrs));
}

int main()
{

    vector<double> p_value = {1,	0,1,	0,0.5,	0,0.5,	0,1,	0,1,	0,0,	0,0,	0,0.25,	0.25,0.25,	0.25,0.5,	0.5,0.5,	0.5,1,	0,1,	0,0.5,	0,0.5,	0,1,	0,1,	0 };

    vector<size_t> p_index = { 0, 8, 1, 8, 2, 8, 3, 8, 2, 8, 3, 8, 8, 8, 8, 8, 2, 6, 3, 7, 2, 6, 3, 7, 4, 8, 5, 8, 6, 8, 7, 8, 6, 8, 7, 8};

    vector<double> value = { 1,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0, 1,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0, 830771.2,	0.000000000029802304,	-253845.76,	-11538.432,	-415384.96,	0.000000000029802304,	46153.92,	11538.432,	0,	0,	0,	0, 0.000000000059604672,	830771.2,	11538.432,	46153.92,	0.000000000059604672,	-415384.96,	-11538.432,	-253845.76,	0,	0,	0,	0, -253845.76,	11538.432,	415384.32,	-150000,	46153.92,	-11538.432,	-207692.16,	150000,	0,	0,	0,	0, -11538.432,	46153.92,	-150000,	415384.32,	11538.432,	-253845.76,	150000,	-207692.16,	0,	0,	0,	0, 1,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0, 1,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0, -415384.96,	0.000000000029802304,	46153.92,	11538.432,	1661536,	0.000000000059604672,	-507692.16,	-23076.928,	-415384.96,	0.000000000029802304,	46153.92,	11538.432, 0.000000000029802304,	-415384.96,	-11538.432,	-253845.76,	0.00000000008940672,	1661536,	23076.928,	92307.84,	0.000000000059604672,	-415384.96,	-11538.432,	-253845.76, 46153.92,	-11538.432,	-207692.16,	150000,	-507692.16,	23076.928,	830771.2,	-300000,	46153.92,	-11538.432,	-207692.16,	150000, 11538.432,	-253845.76,	150000,	-207692.16,	-23076.928,	92307.84,	-300000,	830771.2,	11538.432,	-253845.76,	150000,	-207692.16, 1,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0, 1,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0, -415384.96,	0.000000000029802304,	46153.92,	11538.432,	830771.2,	0.000000000029802304,	-253845.76,	-11538.432,	0,	0,	0,	0, 0.000000000029802304,	-415384.96,	-11538.432,	-253845.76,	0.000000000029802304,	830771.2,	11538.432,	46153.92,	0,	0,	0,	0, 46153.92,	-11538.432,	-207692.16,	150000,	-253845.76,	11538.432,	415384.32,	-150000,	0,	0,	0,	0, 11538.432,	-253845.76,	150000,	-207692.16,	-11538.432,	46153.92,	-150000,	415384.32,	0,	0,	0,	0,};

    vector<size_t> index = { 0,	18,	18,	18,	18,	18,	18,	18,	18,	18,	18,	18, 1,	18,	18,	18,	18,	18,	18,	18,	18,	18,	18,	18, 2,	4,	5,	8,	10,	11,	18,	18,	18,	18,	18,	18, 3,	4,	5,	9,	10,	11,	18,	18,	18,	18,	18,	18, 2,	3,	4,	5,	8,	9,	10,	11,	18,	18,	18,	18, 2,	3,	4,	5,	8,	9,	10,	11,	18,	18,	18,	18, 6,	18,	18,	18,	18,	18,	18,	18,	18,	18,	18,	18, 7,	18,	18,	18,	18,	18,	18,	18,	18,	18,	18,	18, 2,	4,	5,	8,	10,	11,	14,	16,	17,	18,	18,	18, 3,	4,	5,	9,	10,	11,	15,	16,	17,	18,	18,	18, 2,	3,	4,	5,	8,	9,	10,	11,	14,	15,	16,	17, 2,	3,	4,	5,	8,	9,	10,	11,	14,	15,	16,	17, 12,	18,	18,	18,	18,	18,	18,	18,	18,	18,	18,	18, 13,	18,	18,	18,	18,	18,	18,	18,	18,	18,	18,	18, 8,	10,	11,	14,	16,	17,	18,	18,	18,	18,	18,	18, 9,	10,	11,	15,	16,	17,	18,	18,	18,	18,	18,	18, 8,	9,	10,	11,	14,	15,	16,	17,	18,	18,	18,	18, 8,	9,	10,	11,	14,	15,	16,	17,	18,	18,	18,	18};

    vector<size_t> numrows = { 8, 18 };

    //// P
    size_t p_mrs = 2;
    double* d_p_value;
    size_t* d_p_index;

    CUDA_CALL( cudaMalloc( (void**)&d_p_value, sizeof(double) * numrows[1]*p_mrs) );
    CUDA_CALL( cudaMemcpy(d_p_value, &p_value[0], sizeof(double) * numrows[1]*p_mrs, cudaMemcpyHostToDevice) );
    CUDA_CALL( cudaMalloc( (void**)&d_p_index, sizeof(size_t) * numrows[1]*p_mrs) );
    CUDA_CALL( cudaMemcpy(d_p_index, &p_index[0], sizeof(size_t) * numrows[1]*p_mrs, cudaMemcpyHostToDevice) );

    // A
    size_t mrs = 18;
    double* d_value;
    size_t* d_index;

    CUDA_CALL( cudaMalloc( (void**)&d_value, sizeof(double) * numrows[1]*mrs) );
    CUDA_CALL( cudaMemcpy(d_value, &value[0], sizeof(double) * numrows[1]*mrs, cudaMemcpyHostToDevice) );
    CUDA_CALL( cudaMalloc( (void**)&d_index, sizeof(size_t) * numrows[1]*mrs) );
    CUDA_CALL( cudaMemcpy(d_index, &index[0], sizeof(size_t) * numrows[1]*mrs, cudaMemcpyHostToDevice) );

    // A_ (coarse)
    size_t a_mrs = 4;
    double* d_a_value;
    size_t* d_a_index;

    CUDA_CALL( cudaMalloc( (void**)&d_a_value, sizeof(double) * numrows[0] * a_mrs ) );
    CUDA_CALL( cudaMemset( d_a_value, 0, sizeof(double) * numrows[0] * a_mrs) );
    CUDA_CALL( cudaMalloc( (void**)&d_a_index, sizeof(size_t) * numrows[0] * a_mrs) );
    CUDA_CALL( cudaMemset( d_a_index, 0, sizeof(size_t) * numrows[0] * a_mrs) );

    PTAP_GPU<<<1,1>>>( d_value, d_index, mrs, d_p_value, d_p_index, p_mrs, d_a_value, d_a_index, a_mrs );

    cudaDeviceSynchronize();


}