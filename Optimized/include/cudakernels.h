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

// Determines 2-dimensional CUDA block and grid sizes based on the number of rows N
__host__ void calculateDimensions2D(size_t Nx, size_t Ny, dim3 &gridDim, dim3 &blockDim);

__host__ size_t calcDOF(size_t Nx, size_t Ny, size_t dim);

// returns value of an ELLPack matrix A at (x,y)
__device__ double valueAt(size_t x, size_t y, double* vValue, size_t* vIndex, size_t max_row_size);

// adds the value to an ELLPack matrix A at (x,y)
__device__ void addAt( size_t x, size_t y, double* vValue, size_t* vIndex, size_t max_row_size, double value );


// sets the value of an ELLPack matrix A at (x,y)
__device__ void setAt( size_t x, size_t y, double* vValue, size_t* vIndex, size_t max_row_size, double value );


// specific for assembling restriction matrix
__device__ void setAt_RestMatrix( size_t x, size_t y, double* vValue, size_t* vIndex, size_t num_cols, size_t max_row_size, double value );

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

__global__ void printLinearVector_GPU(size_t* x, size_t i, size_t num_rows, size_t num_cols);
__global__ void printLinearVector_GPU(double* x, size_t i, size_t num_rows, size_t num_cols);

__host__ void printLinearVector(size_t* x, size_t num_rows, size_t num_cols);
__host__ void printLinearVector(double* x, size_t num_rows, size_t num_cols);

__global__ void printVector_GPU(double* x, size_t num_rows);

__global__ void printVector_GPU(std::size_t* x, size_t num_rows);

__global__ void printVector_GPU(int* x);

__global__ void printELL_GPU(double* value, size_t* index, size_t max_row_size, size_t num_rows, size_t num_cols);

__global__ void printELLrow_GPU(size_t row, double* value, size_t* index, size_t max_row_size, size_t num_rows, size_t num_cols);

__host__ void printELLrow(size_t lev, double* value, size_t* index, size_t max_row_size, size_t num_rows, size_t num_cols);

// (scalar) a = b
__global__ void equals_GPU(double* a, double* b);


__global__ void dotProduct_GPU(double* x, double* a, double* b, size_t num_rows);

__global__ void LastBlockDotProduct(double* dot, double* x, double* y, size_t starting_index);


// dot = a[] * b[]
__host__ void dotProduct(double* dot, double* a, double* b, size_t N, dim3 gridDim, dim3 blockDim);

// x = y / z
__global__ void divide_GPU(double *x, double *y, double *z);

// x += y
__global__ void add_GPU(double *x, double *y);

// x -= y
__global__ void minus_GPU(double *x, double *y);


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

__global__ void applyMatrixBC_GPU_test(double* value, size_t* index, size_t max_row_size, size_t bc_index, size_t num_rows, size_t num_cols);
__global__ void applyMatrixBC_GPU_2(double* value, size_t* index, size_t max_row_size, size_t bc_index, size_t num_rows, size_t num_cols);


__host__ void PTAP(vector<vector<double>> &A_, vector<vector<double>> &A, vector<vector<double>> &P, size_t num_rows, size_t num_rows_ );

// a = b
__global__ void vectorEquals_GPU(double* a, double* b, size_t num_rows);



__device__ double matMul(size_t row, size_t col, 
						 double* A_value, size_t* A_index, size_t A_max_row_size, size_t A_num_rows,
						 double* B_value, size_t* B_index, size_t B_max_row_size, size_t b_num_rows	);


////////////////////////////////////////////
// ASSEMBLER
////////////////////////////////////////////

__global__ void transposeELL( double* A, double* B, size_t num_rows, size_t max_row_size);
__global__ void transposeELL( size_t* A, size_t* B, size_t num_rows, size_t max_row_size);

__host__ vector<vector<size_t>> applyBC( vector<size_t> N, size_t numLevels, size_t bc_case, size_t dim);

void applyLoad(vector<double> &b, vector<size_t> N, size_t numLevels, size_t bc_case, size_t dim, double force);

__global__ void assembleGrid2D_GPU( size_t N, size_t dim, double* d_chi, double* d_A_local, double* d_value, size_t* d_index, size_t max_row_size, size_t num_rows, size_t* node_index, size_t p );

__global__ void applyMatrixBC_GPU_obso(double* value, size_t* index, size_t max_row_size, size_t bc_index, size_t num_rows);

__global__ void applyMatrixBC_GPU(double* value, size_t* index, size_t max_row_size, size_t bc_index, size_t num_rows, size_t num_cols);

__global__ void applyProlMatrixBC_GPU_obso(double* value, size_t* index, size_t max_row_size, size_t bc_index, size_t num_rows, size_t num_cols);
__global__ void applyProlMatrixBC_GPU(double* value, size_t* index, size_t max_row_size, size_t* bc_index, size_t num_rows, size_t num_cols, size_t bc_size);

__host__ size_t getFineNode(size_t coarse_index, vector<size_t> N, size_t dim);

__device__ size_t getFineNode_GPU(size_t index, size_t Nx, size_t Ny, size_t Nz, size_t dim);
__device__ int getCoarseNode_GPU(size_t index, size_t Nx, size_t Ny, size_t Nz, size_t dim);
__device__ int getCoarseNode3D_GPU(size_t index, size_t Nx, size_t Ny, size_t Nz);

// __global__ void fillIndexVectorProl2D_GPU(size_t* p_index, size_t Nx, size_t Ny, size_t p_max_row_size, size_t num_rows, size_t num_cols);
__global__ void fillProlMatrix2D_GPU(double* p_value, size_t* p_index, size_t Nx, size_t Ny, size_t p_max_row_size, size_t num_rows, size_t num_cols);
__global__ void fillProlMatrix3D_GPU(double* p_value, size_t* p_index, size_t Nx, size_t Ny, size_t Nz, size_t p_max_row_size, size_t num_rows, size_t num_cols);

__global__ void fillIndexVectorRest2D_GPU(size_t* r_index, size_t Nx, size_t Ny, size_t r_max_row_size, size_t num_rows, size_t num_cols);
__global__ void fillIndexVectorRest3D_GPU(size_t* r_index, size_t Nx, size_t Ny, size_t Nz, size_t r_max_row_size, size_t num_rows, size_t num_cols);

__global__ void fillProlMatrix(	double* p_value, size_t* p_index, size_t p_max_row_size, size_t num_rows, size_t num_cols, size_t Nx, size_t Ny, size_t Nz, size_t dim);
__global__ void fillRestMatrix(double* r_value, size_t* r_index, size_t r_max_row_size, double* p_value, size_t* p_index, size_t p_max_row_size, size_t num_rows, size_t num_cols);

__global__ void fillIndexVector2D_GPU(size_t* index, size_t Nx, size_t Ny, size_t max_row_size, size_t num_rows);
__global__ void fillIndexVector3D_GPU(size_t* index, size_t Nx, size_t Ny, size_t Nz, size_t max_row_size, size_t num_rows);


////////////////////////////////////////////
// SMOOTHERS
////////////////////////////////////////////

__global__ void Jacobi_Precond_GPU(double* c, double* value, size_t* index, size_t max_row_size, double* r, size_t num_rows, double damp);


////////////////////////////////////////////
// SOLVER
////////////////////////////////////////////

__global__ void checkIterationConditions(bool* foo, size_t* step, double* res, double* res0, double* m_minRes, double* m_minRed, size_t m_maxIter);
__global__ void checkIterationConditionsBS(bool* foo, size_t* step, size_t m_maxIter, double* res, double* m_minRes);

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

__global__ void UpdateDrivingForce(double *df, double* uTau, double p, double *chi, double local_volume, size_t N);

__global__ void calcDrivingForce_GPU(double *x, double *u, double* chi, double p, size_t *node_index, double* d_A_local, size_t num_rows, size_t dim, double local_volume);

__host__ void calcDrivingForce(double *df, double *chi, double p, double *temp, double *u, vector<size_t*> node_index, double* d_A_local, size_t num_rows, dim3 gridDim, dim3 blockDim, size_t dim, size_t numElements, double local_volume);

//TODO: delete
__host__ void TestcalcDrivingForce(double *df, double *chi, double p, double *u, size_t* node_index, double* d_A_local, size_t num_rows, dim3 gridDim, dim3 blockDim, size_t numElements);

//TODO: delete
__global__ void calcDrivingForce_(double *df, double *chi, double p, double *temp, double *u, size_t* node_index, double* d_A_local, size_t num_rows, size_t dim);

__global__ void sumOfVector_GPU(double* sum, double* x, size_t n);

__device__ double laplacian_GPU( double *array, size_t ind, size_t Nx, size_t Ny, size_t Nz, double h );

__global__ void calcLambdaUpper(double *df_array, double *max, int *mutex, double* beta, double *chi, double* eta, int Nx, int Ny, int Nz, unsigned int numElements, double h);

__global__ void calcLambdaLower(double *df_array, double *min, int *mutex, double* beta, double *chi, double* eta, int Nx, int Ny, int Nz, unsigned int numElements, double h);

__global__ void calcChiTrial( double *chi, double *df, double *lambda_trial, double del_t, double* eta, double* beta, double* chi_trial, size_t Nx, size_t Ny, size_t Nz, size_t numElements, double h);

__global__ void calcLambdaTrial(double *rho_trial, double rho, double *lambda_l, double *lambda_u, double *lambda_trial);

__global__ void int_g_p(double* d_temp, double* d_df, double local_volume, size_t numElements);

__global__ void calcP_w_GPU(double* p_w, double* df, double* uTAu, double* chi, int p, double local_volume, size_t numElements);

__host__ void calcP_w(double* p_w, double* sum_g, double* sum_df_g, double* df, double* chi, double* g, double* df_g, size_t numElements, double local_volume);

__global__ void calcSum_df_g_GPU(double* sum, double* df, double* g, size_t numElements);

__global__ void calcEtaBeta( double* eta, double* beta, double etastar, double betastar, double* p_w );

__global__ void calcRhoTrial(double* m_d_rho_tr, double local_volume, size_t numElements);

__global__ void checkTDOConvergence(bool* foo, double rho, double* rho_trial);

__global__ void RA(vector<double*> r_value, vector<size_t*> r_index, vector<size_t> r_max_row_size, size_t num_rows, size_t num_cols);

__global__ void AP(	double* value, size_t* index, size_t max_row_size, double* p_value, size_t* p_index, size_t p_max_row_size, double* temp_matrix, size_t num_rows, size_t num_cols);

__host__ void RAP(	vector<double*> value, vector<size_t*> index, vector<size_t> max_row_size, 
					vector<double*> r_value, vector<size_t*> r_index, vector<size_t> r_max_row_size, 
					vector<double*> p_value, vector<size_t*> p_index, vector<size_t> p_max_row_size, 
					double* temp_matrix,
					vector<size_t> num_rows, 
					size_t lev);

__global__ void RAP_(	double* value, size_t* index, size_t max_row_size, size_t num_rows,
						double* value_, size_t* index_, size_t max_row_size_, size_t num_rows_, 
						double* r_value, size_t* r_index, size_t r_max_row_size, 
						double* p_value, size_t* p_index, size_t p_max_row_size, 
						size_t lev);

// DEBUG: TEMP:
__global__ void checkLaplacian(double* laplacian, double* chi, size_t Nx, size_t Ny, size_t Nz, size_t numElements, double h);



// DEBUG:
__global__ void checkMassConservation(double* chi, double local_volume, size_t numElements);

__global__ void bar(double* x);



// transposed ELL
__device__ void addAt_( size_t x, size_t y, double* vValue, size_t* vIndex, size_t max_row_size, size_t num_rows, double value );
__device__ void setAt_( size_t x, size_t y, double* vValue, size_t* vIndex, size_t max_row_size, size_t num_rows, double value );
__global__ void printELL_GPU_(double* value, size_t* index, size_t max_row_size, size_t num_rows, size_t num_cols);

__global__ void printELLrow_GPU_(size_t row, double* value, size_t* index, size_t max_row_size, size_t num_rows, size_t num_cols);

__host__ void printELLrow_(size_t lev, double* value, size_t* index, size_t max_row_size, size_t num_rows, size_t num_cols);
__device__ double valueAt_(size_t x, size_t y, double* vValue, size_t* vIndex, size_t max_row_size, size_t num_rows);
__global__ void assembleGrid2D_GPU_( size_t N, size_t dim, double* d_chi, double* d_A_local, size_t num_rows_l, double* d_value, size_t* d_index, size_t max_row_size, size_t num_rows, size_t* node_index, size_t p );
__global__ void Apply_GPU_ (const std::size_t num_rows, const std::size_t max_row_size,const double* value,const std::size_t* index,const double* x,double* r);
__global__ void applyMatrixBC_GPU_(double* value, size_t* index, size_t max_row_size, size_t bc_index, size_t num_rows, size_t num_cols);

#endif // CUDAKERNELS_H

