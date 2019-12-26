#include <iostream>
#include <cuda.h>
#include <vector>
#include <cuda_runtime.h>
// #include "../include/mycudaheader.h"
// #include "precond.h"

using namespace std;


__global__ 
void Jacobi_Precond_GPU(double* c, double* value, double* r, size_t num_rows)
{

	int id = blockDim.x * blockIdx.x + threadIdx.x;

	if ( id < num_rows )
		c[id] = value[id] * r[id];
}