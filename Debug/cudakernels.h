#ifndef CUDAKERNELS_H
#define CUDAKERNELS_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include <ctime>
#include <iostream>


#define CUDA_CALL( call )                                                                                          \
    {                                                                                                                  \
    cudaError_t err = call;                                                                                          \
    if ( cudaSuccess != err)                                                                                         \
        fprintf(stderr, "CUDA error for %s in %d of %s : %s.\n", #call , __LINE__ , __FILE__ ,cudaGetErrorString(err));\
    }


using namespace std;


// Self-defined double-precision atomicAdd function for nvidia GPUs with Compute Capability 6 and below.
// Pre-defined atomicAdd() with double-precision does not work for pre-CC7 nvidia GPUs.
__device__ double atomicAdd_double(double* address, double val);

// TODO: to repair
// Determines 1-dimensional CUDA block and grid sizes based on the number of rows N
__host__ void calculateDimensions(size_t N, dim3 &gridDim, dim3 &blockDim);

__host__ size_t calcDOF(size_t Nx, size_t Ny, size_t dim);

// returns value of an ELLPack matrix A at (x,y)
__device__ double valueAt(size_t x, size_t y, double* vValue, size_t* vIndex, size_t max_row_size);

// sets the value of an ELLPack matrix A at (x,y)
__device__ void setAt( size_t x, size_t y, double* vValue, size_t* vIndex, size_t max_row_size, double value );

__global__ void setToZero(double* a, size_t num_rows);

// norm = x.norm()
__global__ void norm_GPU(double* norm, double* x, size_t num_rows);

// a[] = 0


// a[] = 0, size_t
__global__ void setToZero(size_t* a, size_t num_rows);

//TODO: to delete
// bool = true
__global__ void setToTrue( bool *foo );


// DEBUG: TEST !!!!!!!!!!!!!!!!!!!!!!!!!!
__global__ void sqrt_GPU(double *x);

// sum = sum( x[n]*x[n] )
__global__ void sumOfSquare_GPU(double* sum, double* x, size_t n);


__global__ void LastBlockSumOfSquare_GPU(double* sum, double* x, size_t n, size_t counter);

__host__ void norm_GPU(double* d_norm, double* d_x, size_t N, dim3 gridDim, dim3 blockDim);


/// Helper functions for debugging
__global__ void print_GPU(double* x);

__global__ void print_GPU(int* x);

__global__ void print_GPU(size_t* x);

__global__ void print_GPU(bool* x);

__global__ void printVector_GPU(double* x);

__global__ void printVector_GPU(double* x, size_t num_rows);

__global__ void printVector_GPU(std::size_t* x, size_t num_rows);

__global__ void printVector_GPU(int* x);

// (scalar) a = b
__global__ void equals_GPU(double* a, double* b);


// x = a * b
__global__ void dotProduct(double* x, double* a, double* b, size_t num_rows);


__global__ void LastBlockDotProduct(double* dot, double* x, double* y, size_t starting_index);


// dot = a[] * b[]
__host__ void dotProduct_test(double* dot, double* a, double* b, size_t N, dim3 gridDim, dim3 blockDim);

// x = y / z
__global__ void divide_GPU(double *x, double *y, double *z);
 


// x += c
__global__ void addVector_GPU(double *x, double *c, size_t num_rows);


__global__ void transformToELL_GPU(double *array, double *value, size_t *index, size_t max_row_size, size_t num_rows);


std::size_t getMaxRowSize(std::vector<double> &array, std::size_t num_rows, std::size_t num_cols);

// transforms a flattened matrix (array) to ELLPACK's vectors value and index
// max_row_size has to be d prior to this
void transformToELL(std::vector<double> &array, double* value, size_t* index, size_t max_row_size, size_t num_rows);
// void transformToELL(std::vector<double> &array, std::vector<double> &value, std::vector<std::size_t> &index, size_t max_row_size, size_t num_rows);

// sets identity rows and columns of the DOF in which a BC is applied
void applyMatrixBC(double *array, size_t index, size_t num_rows, size_t num_cols);


// a = b
__global__ void vectorEquals_GPU(double* a, double* b, size_t num_rows);


////////////////////////////////////////////
// SMOOTHERS
////////////////////////////////////////////

__global__ void Jacobi_Precond_GPU(double* c, double* value, size_t* index, size_t max_row_size, double* r, size_t num_rows);


////////////////////////////////////////////
// SOLVER
////////////////////////////////////////////


__global__ void printInitialResult_GPU(double* res0, double* m_minRes, double* m_minRed);

/// r = b - A*x
__global__ void ComputeResiduum_GPU(const std::size_t num_rows, const std::size_t num_cols_per_row,const double* value,const std::size_t* index,const double* x,double* r,double* b);

/// r = r - A*x
__global__ void UpdateResiduum_GPU( const std::size_t num_rows, const std::size_t num_cols_per_row, const double* value, const std::size_t* index, const double* x, double* r);




#endif // CUDAKERNELS_H