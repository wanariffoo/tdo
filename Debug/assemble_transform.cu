/*
NOTE:

    pseudocode:

    insert N, which is the number of elements per row
    calculate each element's local stiffness matrix (include the design variable (TDO's density))
    assemble a global stiffness matrix from N, in COO
    convert COO to ELLPACK

    solve for u

    



*/




#include <iostream>
#include <cuda.h>
#include <vector>
#include <cuda_runtime.h>
#include "../include/mycudaheader.h"

using namespace std;


__global__ 
void getMaxRowSize(double *array, size_t *max, int *mutex, size_t n)
{
	unsigned int id = threadIdx.x + blockIdx.x*blockDim.x;
	unsigned int stride = gridDim.x*blockDim.x;

    __shared__ size_t local_nnz[1024];

    local_nnz[id] = 0;


    // CHECK: something's fishy here
    // get nnz of each row
        for ( int j = 0 ; j < stride ; j++)
        {

            if ( array[j + stride*id] != 0)
            local_nnz[id]++;
        }

	// reduction
	unsigned int i = blockDim.x/2;
    while(i != 0)
    {
		if(threadIdx.x < i){
			local_nnz[threadIdx.x] = fmaxf(local_nnz[threadIdx.x], local_nnz[threadIdx.x + i]);
		}
		__syncthreads();
		i /= 2;
	}

    if(threadIdx.x == 0)
    {
		while(atomicCAS(mutex,0,1) != 0);  //lock
		*max = fmaxf(*max, local_nnz[0]);
        atomicExch(mutex, 0);  //unlock
    }
}




void transformToELL(double *array, double *value, double *index, size_t max_row_size, size_t num_rows)
{
    size_t counter = 0;

    for ( int i = 0 ; i < num_rows ; i++ )
    {
        size_t nnz = 0;

        for ( int j = 0 ; j < num_rows && nnz < max_row_size ; j++ )
        {
            if ( array [ j + i*num_rows ] != 0 )
            {
                value [counter] = array [ j + i*num_rows ];
                index [counter] = j;
                counter++;
                nnz++;
            }

            if ( j == num_rows - 1 )
            {
                for ( int i = counter ; nnz < max_row_size ; counter++ && nnz++ )
                {
                    value [counter] = 0.0;
                    index [counter] = num_rows;
                }
            }
        }
    }
}


int main()
{
                                                    

    size_t max_row_size = 0;
    size_t num_rows = 18;

    // vector<double> array = {4, -1, 0, 0, -1, 4, -1, 0, 0, -1, 4, -1, 0, 0, -1, 4};
    vector<double> array = {4, 	1, 	0, 	0, 	0, 	0, 	0, 	0, 	0, 	0, 	0, 	0, 	0, 	0, 	0, 	0, 	0, 	0, \
                            1, 	4, 	1, 	0, 	0, 	0, 	0, 	0, 	0, 	0, 	0, 	0, 	0, 	0, 	0, 	0, 	0, 	0, \
                            0, 	1, 	8, 	2, 	0, 	0, 	0, 	0, 	0, 	0, 	0, 	0, 	0, 	0, 	0, 	0, 	0, 	0, \
                            0, 	0, 	2, 	8, 	1, 	0, 	1, 	0, 	0, 	0, 	0, 	0, 	0, 	0, 	0, 	0, 	0, 	0, \
                            0, 	0, 	0, 	1, 	4, 	1, 	0, 	0, 	0, 	0, 	0, 	0, 	0, 	0, 	0, 	0, 	0, 	0, \
                            0, 	0, 	0, 	0, 	1, 	4, 	0, 	0, 	5, 	1, 	0, 	0, 	0, 	0, 	0, 	0, 	0, 	0, \
                            0, 	0, 	0, 	1, 	0, 	0, 	8, 	2, 	1, 	4, 	1, 	0, 	0, 	0, 	0, 	0, 	0, 	0, \
                            0, 	0, 	0, 	0, 	0, 	0, 	2, 	8, 	2, 	1, 	4, 	1, 	0, 	0, 	0, 	0, 	0, 	0, \
                            0, 	0, 	0, 	0, 	0, 	1, 	0, 	2, 	12, 3, 	1, 	4, 	0, 	0, 	0, 	0, 	0, 	0, \
                            0, 	0, 	0, 	0, 	0, 	0, 	0, 	0, 	4, 	16, 2, 	0, 	1, 	0, 	0, 	0, 	0, 	0, \
                            0, 	0, 	0, 	0, 	0, 	0, 	0, 	0, 	0, 	2, 	8, 	2, 	0, 	0, 	0, 	0, 	0, 	0, \
                            0, 	0, 	0, 	0, 	0, 	0, 	0, 	0, 	0, 	0, 	2, 	8, 	0, 	0, 	1, 	0, 	0, 	0, \
                            0, 	0, 	0, 	0, 	0, 	0, 	0, 	0, 	0, 	1, 	0, 	0, 	4, 	1, 	0, 	0, 	0, 	0, \
                            0, 	0, 	0, 	0, 	0, 	0, 	0, 	0, 	0, 	0, 	0, 	0, 	1, 	4, 	1, 	0, 	0, 	0, \
                            0, 	0, 	0, 	0, 	0, 	0, 	0, 	0, 	0, 	0, 	0, 	1, 	0, 	1, 	8, 	2, 	0, 	0, \
                            4, 	1, 	0, 	0, 	0, 	0, 	0, 	0, 	0, 	0, 	0, 	0, 	0, 	0, 	2, 	8, 	1, 	0, \
                            1, 	4, 	5, 	1, 	0, 	0, 	0, 	0, 	0, 	0, 	0, 	0, 	0, 	0, 	0, 	1, 	4, 	1, \
                            0, 	1, 	5, 	5, 	1, 	0, 	0, 	0, 	0, 	0, 	0, 	0, 	0, 	0, 	0, 	0, 	1, 	4, };

    // CUDA

    double *d_array = nullptr;
    double *d_max_row_size = nullptr;
    int *d_mutex = nullptr;
    size_t *d_max = nullptr;


    CUDA_CALL( cudaMalloc( (void**)&d_mutex, sizeof(int) ) );
    CUDA_CALL( cudaMalloc( (void**)&d_max, sizeof(size_t) ) );
    CUDA_CALL( cudaMalloc( (void**)&d_array, sizeof(double) * num_rows * num_rows ) );

    CUDA_CALL( cudaMalloc( (void**)&d_max_row_size, sizeof(double) ) );
    
    CUDA_CALL( cudaMemset(d_max, 0, sizeof(size_t)) );
    CUDA_CALL( cudaMemset(d_mutex, 0, sizeof(int)) );

    CUDA_CALL( cudaMemcpy(d_array, &array[0], sizeof(double) * num_rows * num_rows , cudaMemcpyHostToDevice) ); 



    CUDA_CALL( cudaMemcpy(d_max_row_size, &max_row_size, sizeof(double), cudaMemcpyHostToDevice) );


    getMaxRowSize<<<1,num_rows>>>(d_array, d_max, d_mutex, num_rows);
    cudaDeviceSynchronize();

    print_GPU<<<1,1>>>(d_max);
    cudaDeviceSynchronize();

    // // initialize the value and index vectors

    // vector<double> value;
    // vector<double> index;

    // value.resize(max_row_size * num_rows);
    // index.resize(max_row_size * num_rows);

    // transformToELL( &array[0], &value[0], &index[0], max_row_size, num_rows );

    // cout << "value : ";
    // for ( int i = 0 ; i < value.size() ; i ++ )
    //     cout << value[i] << " ";
    // cout << "\n";

    // cout << "index : ";
    // for ( int i = 0 ; i < value.size() ; i ++ )
    //     cout << index[i] << " ";
    // cout << "\n";

    // cout << "global : ";
    // for ( int j = 0 ; j < num_rows ; j ++ )
    // {
    //     for ( int i = 0 ; i < num_rows ; i ++ )
    //        cout << array[i + j*num_rows] << " ";
        
           
    //     cout << "\n";
    // }


}