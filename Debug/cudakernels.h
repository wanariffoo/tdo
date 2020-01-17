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

// adds the value to an ELLPack matrix A at (x,y)
__device__ void addAt( size_t x, size_t y, double* vValue, size_t* vIndex, size_t max_row_size, double value );

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

__global__ void printELL_GPU(double* value, size_t* index, size_t max_row_size, size_t num_rows, size_t num_cols);

// (scalar) a = b
__global__ void equals_GPU(double* a, double* b);


__global__ void dotProduct_GPU(double* x, double* a, double* b, size_t num_rows);

__global__ void LastBlockDotProduct(double* dot, double* x, double* y, size_t starting_index);


// dot = a[] * b[]
__host__ void dotProduct(double* dot, double* a, double* b, size_t N, dim3 gridDim, dim3 blockDim);

// x = y / z
__global__ void divide_GPU(double *x, double *y, double *z);
 


// x += c
__global__ void addVector_GPU(double *x, double *c, size_t num_rows);



__global__ void transformToELL_GPU(double *array, double *value, size_t *index, size_t max_row_size, size_t num_rows);


std::size_t getMaxRowSize(vector<vector<double>> &array, size_t num_rows, size_t num_cols);

// transforms a flattened matrix (array) to ELLPACK's vectors value and index
// max_row_size has to be d prior to this
void transformToELL(vector<vector<double>> &array, vector<double> &value, vector<size_t> &index, size_t max_row_size, size_t num_rows, size_t num_cols );
// void transformToELL(std::vector<double> &array, std::vector<double> &value, std::vector<std::size_t> &index, size_t max_row_size, size_t num_rows);

// sets identity rows and columns of the DOF in which a BC is applied
void applyMatrixBC(vector<vector<double>> &array, size_t index, size_t num_rows, size_t dim);

__host__ void PTAP(vector<vector<double>> &A_, vector<vector<double>> &A, vector<vector<double>> &P, size_t num_rows, size_t num_rows_ );

// a = b
__global__ void vectorEquals_GPU(double* a, double* b, size_t num_rows);


////////////////////////////////////////////
// ASSEMBLER
////////////////////////////////////////////

__global__ void assembleGrid2D_GPU( size_t N, size_t dim, double* d_kai, double* d_A_local, double* d_value, size_t* d_index, size_t max_row_size, size_t num_rows, size_t* node_index, size_t p );

__global__ void applyMatrixBC_GPU(double* value, size_t* index, size_t max_row_size, size_t bc_index, size_t num_rows);

__host__ size_t getFineNode(size_t coarse_index, vector<size_t> N, size_t dim);

////////////////////////////////////////////
// SMOOTHERS
////////////////////////////////////////////

__global__ void Jacobi_Precond_GPU(double* c, double* value, size_t* index, size_t max_row_size, double* r, size_t num_rows, double damp);


////////////////////////////////////////////
// SOLVER
////////////////////////////////////////////


__global__ void printInitialResult_GPU(double* res0, double* m_minRes, double* m_minRed);

/// r = b - A*x
__global__ void ComputeResiduum_GPU(const std::size_t num_rows, const std::size_t num_cols_per_row,const double* value,const std::size_t* index,const double* x,double* r,double* b);

/// r = r - A*x
__global__ void UpdateResiduum_GPU( const std::size_t num_rows, const std::size_t num_cols_per_row, const double* value, const std::size_t* index, const double* x, double* r);

__global__ void Apply_GPU(const std::size_t num_rows, const std::size_t num_cols_per_row, const double* value, const std::size_t* index, const double* x, double* r);

__global__ void ApplyTransposed_GPU( const std::size_t num_rows, const std::size_t num_cols_per_row, const double* value, const std::size_t* index, const double* x, double* r);

__global__ void printResult_GPU(size_t* step, double* res, double* m_minRes, double* lastRes, double* res0, double* m_minRed);

__global__ void addStep(size_t* step);

////////////////////////////////////////////
// BASE SOLVER
////////////////////////////////////////////

__global__ void calculateDirectionVector(	 size_t* d_step, double* d_p,  double* d_z,  double* d_rho,  double* d_rho_old, size_t num_rows);

__host__ void calculateAlpha(double* d_alpha, double* d_rho, double* d_p, double* d_z, double* d_alpha_temp, size_t num_rows, dim3 gridDim, dim3 blockDim);


// x = x + alpha * p
__global__ void axpy_GPU(double* d_x, double* d_alpha, double* d_p, size_t num_rows);

// x = x - alpha * p
__global__ void axpy_neg_GPU(double* d_x, double* d_alpha, double* d_p, size_t num_rows);

////////////////////////////////////////////
// TDO
////////////////////////////////////////////

__global__ void UpdateDrivingForce(double *df, double* uTau, double p, double *kai, double local_volume, size_t N);

__global__ void uTAu_GPU(double *x, double *u, size_t *node_index, double* d_A_local, size_t num_rows);

__host__ void calcDrivingForce(double *df, double *kai, double p, double *temp, double *u, size_t* node_index, double* d_A_local, size_t num_rows, dim3 gridDim, dim3 blockDim);

__global__ void sumOfVector_GPU(double* sum, double* x, size_t n);

__device__ double laplacian_GPU( double *array, size_t ind, size_t N );

__global__ void calcLambdaUpper(double *df_array, double *max, int *mutex, double* beta, double *kai, double* eta, unsigned int N, unsigned int numElements);

__global__ void calcLambdaLower(double *df_array, double *min, int *mutex, double* beta, double *kai, double* eta, unsigned int N, unsigned int numElements);

__global__ void calcKaiTrial( double *kai, double *df, double *lambda_trial, double del_t, double* eta, double* beta, double* kai_trial, size_t N, size_t numElements);

__global__ void calcLambdaTrial(double *rho_trial, double rho, double *lambda_l, double *lambda_u, double *lambda_trial);

__global__ void int_g_p(double* d_temp, double* d_df, double local_volume, size_t numElements);

__global__ void calcP_w(double* p_w, double* df, double* uTAu, double* kai, int p, double local_volume, size_t numElements);

__global__ void calcEtaBeta( double* eta, double* beta, double etastar, double betastar, double* p_w );


// DEBUG:

__global__ void temp(double* d_kai);

#endif // CUDAKERNELS_H