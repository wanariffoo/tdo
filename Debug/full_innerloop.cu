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


int main()
{
    size_t num_rows = 8;
    size_t num_GP = 4;
    size_t max_row_size = 8;
    size_t N = 2;

    // rho
    double rho = 0.4;

    // displacement vector
    vector<double> u = {0, 0, 0.00010427, 8.3599E-05, 0.00010385, 0.00018609, 0, 0, 2.0302E-07, 8.3438E-05, 1.873E-07, 0.00018757, 0, 0, -0.00010436, 8.34E-05, -0.00010443, 0.00018798};


    vector<double> temp(num_rows, 0.0);
    
    // double *d_u;

    // CUDA_CALL ( cudaMalloc( (void**)&d_u, sizeof(double) * num_rows) );
    // CUDA_CALL ( cudaMemcpy( d_u, &u[0], sizeof(double) * num_rows, cudaMemcpyHostToDevice) );


    // inner loop
    double eta = 12;
    double beta = 1;
    double h = 0.5;

    // driving force
    // double kai = 0.4;
    // double df;
    vector<double> p(0, num_GP);
    
    // bisection
    double del_t = 1;
    double lambda_trial = 0;
    double lambda_min;
    double lambda_max;



    vector<double> l_value = {
        103939100,	37502100,	-63536480,	-2900100,	-51968000,	-37502400,	11566200,	2900100,
        37502100,	103939100,	2900100,	11566200,	-37502400,	-51968000,	-2900100,	-63536480,
        -63536480,	2900100,	103939100,	-37502100,	11566200,	-2900100,	-51968000,	37502400,
        -2900100,	11566200,	-37502100,	103939100,	2900100,	-63536480,	37502400,	-51968000,
        -51968000,	-37502400,	11566200,	2900100,	103939100,	37502100,	-63536480,	-2900100,
        -37502400,	-51968000,	-2900100,	-63536480,	37502100,	103939100,	2900100,	11566200,
        11566200,	-2900100,	-51968000,	37502400,	-63536480,	2900100,	103939100,	-37502100,
        2900100,	-63536480,	37502400,	-51968000,	-2900100,	11566200,	-37502100,	103939100
    };

    vector<size_t> l_index = {
        0,	1,	2,	3,	4,	5,	6,	7,
        0,	1,	2,	3,	4,	5,	6,	7,
        0,	1,	2,	3,	4,	5,	6,	7,
        0,	1,	2,	3,	4,	5,	6,	7,
        0,	1,	2,	3,	4,	5,	6,	7,
        0,	1,	2,	3,	4,	5,	6,	7,
        0,	1,	2,	3,	4,	5,	6,	7,
        0,	1,	2,	3,	4,	5,	6,	7
    };


    // CUDA

    double *d_eta;
    double *d_n;
    double *d_beta;
    // double *d_kai;
    // double *d_df;
    // double *d_df1;
    // double *d_df2;
    // double *d_df3;

    // double *d_p;
    double *d_temp;
    double *d_u;

    double *d_l_value;
    size_t *d_l_index;

    // bisection

    int *d_mutex;

    CUDA_CALL ( cudaMalloc( (void**)&d_eta, sizeof(double) ) );
    CUDA_CALL ( cudaMalloc( (void**)&d_n, sizeof(double) ) );
    CUDA_CALL ( cudaMalloc( (void**)&d_beta, sizeof(double) ) );
    // CUDA_CALL ( cudaMalloc( (void**)&d_df, sizeof(double) ) );
    // CUDA_CALL ( cudaMalloc( (void**)&d_df1, sizeof(double) ) );
    // CUDA_CALL ( cudaMalloc( (void**)&d_df2, sizeof(double) ) );
    // CUDA_CALL ( cudaMalloc( (void**)&d_df3, sizeof(double) ) );
    // CUDA_CALL ( cudaMalloc( (void**)&d_kai, sizeof(double) ) );
    CUDA_CALL ( cudaMalloc( (void**)&d_temp, sizeof(double) * num_rows) );
    CUDA_CALL ( cudaMalloc( (void**)&d_u, sizeof(double) * 18) );
    CUDA_CALL ( cudaMalloc( (void**)&d_l_value, sizeof(double) * num_rows * max_row_size ) );
    CUDA_CALL ( cudaMalloc( (void**)&d_l_index, sizeof(size_t) * num_rows * max_row_size ) );
    
    CUDA_CALL ( cudaMemset( d_n, 0, sizeof(double) ) );
    // CUDA_CALL ( cudaMemset( d_df, 0, sizeof(double) ) );
    // CUDA_CALL ( cudaMemset( d_df1, 0, sizeof(double) ) );
    // CUDA_CALL ( cudaMemset( d_df2, 0, sizeof(double) ) );
    // CUDA_CALL ( cudaMemset( d_df3, 0, sizeof(double) ) );
    // CUDA_CALL ( cudaMemcpy( d_kai, &kai, sizeof(double), cudaMemcpyHostToDevice) );
    CUDA_CALL ( cudaMemcpy( d_eta, &eta, sizeof(double), cudaMemcpyHostToDevice) );
    CUDA_CALL ( cudaMemcpy( d_beta, &beta, sizeof(double), cudaMemcpyHostToDevice) );
    CUDA_CALL ( cudaMemcpy( d_u, &u[0], sizeof(double) * 18, cudaMemcpyHostToDevice) );
    CUDA_CALL ( cudaMemcpy( d_temp, &temp[0], sizeof(double) * num_rows, cudaMemcpyHostToDevice) );

    CUDA_CALL ( cudaMemcpy( d_l_value, &l_value[0], sizeof(double) * num_rows * max_row_size, cudaMemcpyHostToDevice) );
    CUDA_CALL ( cudaMemcpy( d_l_index, &l_index[0], sizeof(size_t) * num_rows * max_row_size, cudaMemcpyHostToDevice) );

    // node index
    vector<size_t> node_index = {0, 1, 3, 4};
    // size_t* d_node_index = &node_index[0];
    // cudaMalloc( (void**)&d_node_index, sizeof(size_t) * 4 );
    // cudaMemcpy(d_node_index, &node_index[0], sizeof(size_t) * 4, cudaMemcpyHostToDevice);
    

    vector<size_t> node_index1 = {1, 2, 4, 5};
    // size_t* d_node_index1;
    // cudaMalloc( (void**)&d_node_index1, sizeof(size_t) * 4 );
    // cudaMemcpy(d_node_index1, &node_index1[0], sizeof(size_t) * 4, cudaMemcpyHostToDevice);

    vector<size_t> node_index2 = {3, 4, 6, 7};
    // size_t* d_node_index2;
    // cudaMalloc( (void**)&d_node_index2, sizeof(size_t) * 4 );
    // cudaMemcpy(d_node_index2, &node_index2[0], sizeof(size_t) * 4, cudaMemcpyHostToDevice);

    vector<size_t> node_index3 = {4, 5, 7, 8};
    // size_t* d_node_index3;
    // cudaMalloc( (void**)&d_node_index3, sizeof(size_t) * 4 );
    // cudaMemcpy(d_node_index3, &node_index3[0], sizeof(size_t) * 4, cudaMemcpyHostToDevice);


    // vector<double*> df_array;
    // df_array.push_back(d_df);
    // df_array.push_back(d_df1);
    // df_array.push_back(d_df2);
    // df_array.push_back(d_df3);


    vector<double> df = {0, 0, 0, 0};
    double* d_df;
    CUDA_CALL( cudaMalloc( (void**)&d_df, sizeof(double) * 4 ) );
    CUDA_CALL( cudaMemcpy(d_df, &df[0], sizeof(double) * 4, cudaMemcpyHostToDevice) );  

    vector<size_t*> d_node_index;
    d_node_index.resize(4);

    CUDA_CALL( cudaMalloc( (void**)&d_node_index[0], sizeof(size_t) * 4 ) );
    CUDA_CALL( cudaMemcpy(d_node_index[0], &node_index[0], sizeof(size_t) * 4, cudaMemcpyHostToDevice) );  

    CUDA_CALL( cudaMalloc( (void**)&d_node_index[1], sizeof(size_t) * 4 ) );
    CUDA_CALL( cudaMemcpy(d_node_index[1], &node_index1[0], sizeof(size_t) * 4, cudaMemcpyHostToDevice) );  

    CUDA_CALL( cudaMalloc( (void**)&d_node_index[2], sizeof(size_t) * 4 ) );
    CUDA_CALL( cudaMemcpy(d_node_index[2], &node_index2[0], sizeof(size_t) * 4, cudaMemcpyHostToDevice) );  

    CUDA_CALL( cudaMalloc( (void**)&d_node_index[3], sizeof(size_t) * 4 ) );
    CUDA_CALL( cudaMemcpy(d_node_index[3], &node_index3[0], sizeof(size_t) * 4, cudaMemcpyHostToDevice) );  

    vector<double> kai = {0.4,0.4,0.4,0.4};
    double* d_kai;

    CUDA_CALL( cudaMalloc( (void**)&d_kai, sizeof(size_t) * 4 ) );
    CUDA_CALL( cudaMemcpy(d_kai, &kai[0], sizeof(size_t) * 4, cudaMemcpyHostToDevice) );  

    
    double *d_lambda_l;
    double *d_lambda_u;
    double *d_lambda_tr;
    double *d_laplacian;
    vector<double> laplace_array(4); // CHECK: ??

    cudaMalloc( (void**)&d_lambda_l, sizeof(double) );
    cudaMalloc( (void**)&d_lambda_u, sizeof(double) );
    cudaMalloc( (void**)&d_lambda_tr, sizeof(double) );
    cudaMalloc( (void**)&d_laplacian, sizeof(double) * 4 );
    cudaMalloc( (void**)&d_mutex, sizeof(int) );


    cudaMemset( d_lambda_tr, 0, sizeof(double) );
    cudaMemset( d_lambda_u, 0, sizeof(double) );
    cudaMemset( d_lambda_l, 0, sizeof(double) );

    cudaMemcpy(d_laplacian, &laplace_array[0], sizeof(double) * 4, cudaMemcpyHostToDevice);


    double* d_kai_tr;
    cudaMalloc( (void**)&d_kai_tr, sizeof(double) * 4 );
    cudaMemset( d_kai_tr, 0, sizeof(double) * 4);

    //NOTE: reuse this from somewhere?
    double* d_rho_tr;
    cudaMalloc( (void**)&d_rho_tr, sizeof(double));
    cudaMemset( d_rho_tr, 0, sizeof(double));
    
    bool inner_foo = 1;
    bool* d_inner_foo;
    cudaMalloc( (void**)&d_inner_foo, sizeof(bool) );
    cudaMemset( d_inner_foo, 1, sizeof(bool) );
    double volume = 1.0;

    // get block and grid dimensions
    dim3 gridDim;
    dim3 blockDim;
    calculateDimensions( num_rows, gridDim, blockDim );
    
    size_t numElements = 4;


    double* d_uTKu;
    cudaMalloc( (void**)&d_uTKu, sizeof(double) * numElements);
    cudaMemset( d_uTKu, 0, sizeof(double) * numElements);


    ///////////////////////////////////////////////////////////////////////////////////////
    // start inner loop when you have u vector
    ///////////////////////////////////////////////////////////////////////////////////////

    // initialization



    // n is calculated in host
    size_t n_innerloop = (6 / eta) * ( beta / (h*h) );
    // cout << n_innerloop << endl;
    double l_volume = 0.5*0.5;
    // initial driving force
    for ( int i = 0 ; i < numElements ; i++)
    calcDrivingForce ( &d_df[i], &d_kai[i], 3, d_temp, d_u, d_node_index[i], d_l_value, d_l_index, max_row_size, num_rows, gridDim, blockDim );
    cudaDeviceSynchronize();
    
    vectorEquals_GPU<<<1,4>>>(d_uTKu, d_df, 4);
    cudaDeviceSynchronize();


    // printVector_GPU<<<1,4>>>(d_df, 4);
    for ( int j = 0 ; j < n_innerloop; j++ )
    {
        cout << "j = " << j << endl;
        // printVector_GPU<<<1,4>>>(d_kai, 4);
        // cudaDeviceSynchronize();
        
        // ( 1 / 2*element_volume ) * p * pow(kai_element, (p-1) ) * u^T * element_stiffness_matrix * u
        UpdateDrivingForce<<<1,numElements>>> ( d_df, d_uTKu, 3, d_kai, l_volume, numElements);
        
        // printVector_GPU<<<1,4>>>( d_df, 4);
        // cudaDeviceSynchronize();
        
        // bisection algo: 
        
        setToZero<<<1,1>>>(d_lambda_tr, 1);
        calcLambdaUpper<<< 1, 4 >>>(d_df, d_lambda_u, d_mutex, 1.0, d_kai, 12, N, 4);
        calcLambdaLower<<< 1, 4 >>>(d_df, d_lambda_l, d_mutex, 1.0, d_kai, 12, N, 4);
        
        // print_GPU<<<1,1>>> ( d_lambda_l );
        // cudaDeviceSynchronize();
        // print_GPU<<<1,1>>> ( d_lambda_u );
        // cudaDeviceSynchronize();
        
        for ( int i = 0 ; i < 30 ; i++ )
        {
            calcKaiTrial<<<1,4>>> ( d_kai, d_df, d_lambda_tr, del_t, eta, beta, d_kai_tr, 2, numElements);
            setToZero<<<1,1>>>(d_rho_tr, 1);
            sumOfVector_GPU <<< 1, 4 >>> (d_rho_tr, d_kai_tr, 4);
            
            calcLambdaTrial<<<1,1>>>( d_lambda_tr, d_lambda_l, d_lambda_u, d_rho_tr, 0.4, 1.0 );
            // checkRhoTrial<<<1,1>>>( d_inner_foo, d_rho_tr, 0.4 );
        }
        // print_GPU<<<1,1>>>( d_lambda_tr );

        // kai(j) = kai(j+1)
        vectorEquals_GPU<<<1,4>>>( d_kai, d_kai_tr, 4 );

        printVector_GPU<<<1,4>>>( d_kai, 4);
        cudaDeviceSynchronize();

    }
        
    cout << "end of bisection" << endl;
    cudaDeviceSynchronize();

    // update 
}