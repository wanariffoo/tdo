#include <iostream>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>

using namespace std;

#define CUDA_CALL( call )                                                                                          \
    {                                                                                                                  \
    cudaError_t err = call;                                                                                          \
    if ( cudaSuccess != err)                                                                                         \
        fprintf(stderr, "CUDA error for %s in %d of %s : %s.\n", #call , __LINE__ , __FILE__ ,cudaGetErrorString(err));\
    }

__global__ void printLinearVector_GPU(size_t* x, size_t i, size_t num_rows, size_t num_cols)
{
        for ( int j = 0 ; j < num_cols ; j++ )
            printf("%lu ", x[j+i*num_cols]);

        printf("\n");
}

__host__ void printLinearVector(size_t* x, size_t num_rows, size_t num_cols)
{
	for(int i = 0 ; i < num_rows ; i++ )
	{
		printLinearVector_GPU<<<1,1>>>(x, i, num_rows, num_cols);
		cudaDeviceSynchronize();
	}

}

// Determines 1-dimensional CUDA block and grid sizes based on the number of rows N
__host__ 
void calculateDimensions(size_t N, dim3 &gridDim, dim3 &blockDim)
{
    if ( N <= 1024 )
    {
        blockDim.x = 1024; blockDim.y = 1; blockDim.z = 1;
        gridDim.x  = 1; gridDim.y = 1; gridDim.z = 1;
    }
        
    else
    {
        blockDim.x = 1024; blockDim.y = 1; blockDim.z = 1;
        gridDim.x  = (int)ceil(N/blockDim.x)+1; gridDim.y = 1; gridDim.z = 1;
    }
}

/// A*x = r
__global__ 
void Apply_GPU(	
	const std::size_t num_rows, 
	const std::size_t num_cols_per_row,
	const double* value,
	const std::size_t* index,
	const double* x,
	double* r)
{
	int id = blockDim.x * blockIdx.x + threadIdx.x;
	
	if ( id < num_rows )
	{
		double dot = 0;

		for ( int n = 0; n < num_cols_per_row; n++ )
		{
			int col = index [ num_cols_per_row * id + n ];
			double val = value [ num_cols_per_row * id + n ];
			dot += val * x [ col ];
		}
		r[id] = dot;
	}
	
}

// Ax = r
__global__ void Apply_GPU_ (
	const std::size_t num_rows, 
	const std::size_t max_row_size,
	const double* value,
	const std::size_t* index,
	const double* x,
	double* r)
{
    int id = blockDim.x * blockIdx.x + threadIdx.x;

    if ( id < num_rows )
    {
        double sum = 0;
        for ( int n = 0 ; n < max_row_size; n++ )
        {
            unsigned int offset = id + n*num_rows;
            sum += value[offset] * x[index[offset]];
        }
        r[id] = sum;
    }

}


__global__
void printVector_GPU(double* x, size_t num_rows)
{
	int id = blockDim.x * blockIdx.x + threadIdx.x;

	if ( id < num_rows )
		printf("%d %e\n", id, x[id]);
}



// returns value of a transposed ELLPack matrix A at (row,col)
__device__
double valueAt_(size_t row, size_t col, double* vValue, size_t* vIndex, size_t max_row_size, size_t num_rows)
{
    for(size_t k = 0; k < max_row_size; ++k)
    {

        if(vIndex[k * num_rows + row] == col)
                return vValue[k * num_rows + row];

    }

    return 0.0;
}

__global__
void printELL_GPU_(double* value, size_t* index, size_t max_row_size, size_t num_rows, size_t num_cols)
{
	
		for ( int i = 0 ; i < num_rows ; i++)
		{
			for ( int j = 0 ; j < num_cols ; j++)
			printf("%f ", valueAt_(i, j, value, index, max_row_size, num_rows) );

			printf("\n");
		}

	
}


// adds the value to a transposed ELLPack matrix A at (row,col)
__device__
void atomicAddAt_( size_t row, size_t col, double* vValue, size_t* vIndex, size_t max_row_size, size_t num_rows, double value )
{
    for(size_t k = 0; k < max_row_size; ++k)
    {	
		// printf("%d\n", (k * num_rows + y) );
        if(vIndex[k * num_rows + col] == row)
        {
            atomicAdd( &vValue[k * num_rows + col] , value );
            // vValue[k * num_rows + col] += value;
            // printf("%f \n", vValue[k * num_rows + y]);
                k = max_row_size; // to exit for loop
		}
    }
}



// A_coarse = R * A_fine * P
__global__ void PTAP(	double* value, size_t* index, size_t max_row_size, size_t num_rows,
						double* value_, size_t* index_, size_t max_row_size_, size_t num_rows_,
						double* p_value, size_t* p_index, size_t p_max_row_size, 
						size_t lev)
{
    int id = blockDim.x * blockIdx.x + threadIdx.x;

    if( id < num_rows )
    {
        for ( int i_ = 0 ; i_ < p_max_row_size ; i_++ )
        {
            size_t i = p_index[id + i_*num_rows];
            double P_ki = p_value[id + i_*num_rows];
            if(id==0) printf("i = %lu, P_ki = %f\n", i, P_ki);
            // if ( id == 1) printf("%f\n", P_ki);

            for( int l_ = 0 ; l_ < max_row_size ; l_++  )
            {
                size_t l = index[id + l_*num_rows];
                double A_kl = value[id + l_*num_rows];
                double P_ki_A_kl = P_ki * A_kl;
                if(id==0) printf("l = %lu, A_kl = %f\n", l, A_kl);
                if(id==0) printf("P_ki_A_kl = %f\n", P_ki_A_kl);
                for( int j_ = 0 ; j_ < p_max_row_size ; j_++ )
                {
                    size_t j = p_index[l + j_*num_rows];
                    if( j >= num_rows ) break;

                    double P_lj = p_value[l + j_*num_rows];
                    if(id==0) printf("j = %lu, P_lj = %f\n", j, P_lj);
                    double P_ki_A_kl_P_lj = P_ki_A_kl * P_lj;
                    if(id==0) printf("PAP(%lu,%lu) = %f\n", i,j,P_ki_A_kl_P_lj);

                    if(P_ki_A_kl_P_lj != 0.0)
						atomicAddAt_( j, i, value_, index_, max_row_size_, num_rows_, P_ki_A_kl_P_lj );
                }
            }
        }

        // atomicAddAt_( 0, 0, value_, index_, max_row_size_, num_rows_, 10 );
        // atomicAddAt_( 1, 0, value_, index_, max_row_size_, num_rows_, 10 );
        // atomicAddAt_( 0, 2, value_, index_, max_row_size_, num_rows_, 10 );
        // atomicAddAt_( 1, 0, value_, index_, max_row_size_, num_rows_, 10 );
        // atomicAddAt_( 1, 1, value_, index_, max_row_size_, num_rows_, 10 );
        // atomicAddAt_( 1, 2, value_, index_, max_row_size_, num_rows_, 10 );
        // atomicAddAt_( 2, 0, value_, index_, max_row_size_, num_rows_, 10 );
        // atomicAddAt_( 2, 1, value_, index_, max_row_size_, num_rows_, 10 );
        // atomicAddAt_( 2, 2, value_, index_, max_row_size_, num_rows_, 10 );
    }
}

    // for(size_t k = 0; k < P.num_rows(); ++k){
	// 	for(auto it = P.begin(k); it !=  P.end(k); ++it){

	// 		const size_t i = it.index();
	// 		const double& P_ki = it.value();

	// 		for(auto it = A.begin(k); it !=  A.end(k); ++it){

	// 			const size_t l = it.index();
	// 			const double& A_kl = it.value();

	// 			const double P_ki_A_kl = P_ki * A_kl;

	// 			for(auto it = P.begin(l); it !=  P.end(l); ++it){

	// 				const size_t j = it.index();
	// 				const double& P_lj = it.value();

	// 				const double P_ki_A_kl_P_lj =  P_ki_A_kl * P_lj;
	// 				if(P_ki_A_kl_P_lj != 0.0)
	// 					RAP(i,j) += P_ki_A_kl_P_lj;
	// 			}
	// 		}
	// 	}
	// }





int main()
{
    vector<size_t> num_rows = {3,4};
    size_t R_mrs = 2;
    size_t A_mrs = 2;
    size_t P_mrs = 2;

    vector<double> A_value = { 1,1,2,1,5,2,1,0};
    vector<size_t> A_index = { 0,1,1,3,2,3,2,4};
    // vector<double> A_value_ = { 1,1,1,2,2,2,3,3,3};
    vector<size_t> A_index_ = { 0,0,0,1,1,1,2,2,2};
    vector<double> R_value = { 1, 3, 2, 2, 1, 4};
    vector<size_t> R_index = { 0, 3, 1, 2, 1, 3};
    vector<double> P_value = { 1, 2, 2, 3, 0, 1, 0, 4};
    vector<size_t> P_index = { 0,1,1,0,3,2,3,2};
    // vector<double> A_value = { 1, 5, 1, 2, 2, 1, 1, 0};
    // vector<size_t> A_index = { 0, 2, 1, 3, 1, 2, 3, 4};
    // vector<double> P_value = { 1, 0, 2, 1, 2, 0, 3, 4};
    // vector<size_t> P_index = { 0, 3, 1, 2, 1, 3, 0, 2};

    double* d_A_value;
    double* d_A_value_;
    double* d_R_value;
    double* d_P_value;
    size_t* d_A_index;
    size_t* d_A_index_;
    size_t* d_R_index;
    size_t* d_P_index;


    CUDA_CALL( cudaMalloc((void**)&d_A_value, sizeof(double) * num_rows[1] * A_mrs ) );
    CUDA_CALL( cudaMalloc((void**)&d_A_value_, sizeof(double) * num_rows[0] * num_rows[0] ) );
    CUDA_CALL( cudaMalloc((void**)&d_R_value, sizeof(double) * num_rows[0] * R_mrs ) );
    CUDA_CALL( cudaMalloc((void**)&d_P_value, sizeof(double) * num_rows[1] * P_mrs ) );
    CUDA_CALL( cudaMalloc((void**)&d_A_index, sizeof(size_t) * num_rows[1] * A_mrs ) );
    CUDA_CALL( cudaMalloc((void**)&d_A_index_, sizeof(size_t) * num_rows[0] * num_rows[0] ) );
    CUDA_CALL( cudaMalloc((void**)&d_R_index, sizeof(size_t) * num_rows[0] * R_mrs ) ); 
    CUDA_CALL( cudaMalloc((void**)&d_P_index, sizeof(size_t) * num_rows[1] * P_mrs ) );

    CUDA_CALL( cudaMemcpy(d_A_value, &A_value[0], sizeof(double) * num_rows[1] * A_mrs, cudaMemcpyHostToDevice) );
    // CUDA_CALL( cudaMemcpy(d_A_value_, &A_value_[0], sizeof(double) * num_rows[0] * num_rows[0], cudaMemcpyHostToDevice) );
    CUDA_CALL( cudaMemcpy(d_R_value, &R_value[0], sizeof(double) * num_rows[0] * R_mrs, cudaMemcpyHostToDevice) );
    CUDA_CALL( cudaMemcpy(d_P_value, &P_value[0], sizeof(double) * num_rows[1] * P_mrs, cudaMemcpyHostToDevice) );
    CUDA_CALL( cudaMemcpy(d_A_index, &A_index[0], sizeof(size_t) * num_rows[1] * A_mrs, cudaMemcpyHostToDevice) );
    CUDA_CALL( cudaMemcpy(d_A_index_, &A_index_[0], sizeof(size_t) * num_rows[0] * num_rows[0], cudaMemcpyHostToDevice) );
    CUDA_CALL( cudaMemcpy(d_R_index, &R_index[0], sizeof(size_t) * num_rows[0] * R_mrs, cudaMemcpyHostToDevice) );
    CUDA_CALL( cudaMemcpy(d_P_index, &P_index[0], sizeof(size_t) * num_rows[1] * P_mrs, cudaMemcpyHostToDevice) );

    PTAP<<<1,4>>>(d_A_value, d_A_index, A_mrs, num_rows[1], d_A_value_, d_A_index_, 3, num_rows[0], d_P_value, d_P_index, P_mrs, 0);
    cudaDeviceSynchronize();
    
    // printELL_GPU_<<<1,1>>>(d_A_value, d_A_index, A_mrs, num_rows[1], num_rows[1]);
    printELL_GPU_<<<1,1>>>(d_A_value_, d_A_index_, 3, num_rows[0], num_rows[0]);
    // printELL_GPU_<<<1,1>>>(d_P_value, d_P_index, P_mrs, num_rows[1], num_rows[0]);
    // printLinearVector(d_A_index_, 3, 3);


    cudaDeviceSynchronize();

}
