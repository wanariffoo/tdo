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


int main()
{
    // rho
    double rho = 0.4;

    // displacement vector
    vector<double> u = { 0, 0, -0.044797, -0.026485, -0.044798, -0.026487, 0, 0, -0.044798, -0.026485, -0.044797, -0.026485, 0, 0, -0.044798, -0.026486, -0.044799, -0.026485 };

    // inner loop
    double eta = 12;
    double beta = 1;
    double h = 0.5;

    // driving force
    double kai = 0.8;
    
    // bisection
    double del_t = 1;



    // CUDA

    double *d_eta;
    double *d_n;
    double *d_beta;

    cudaMalloc( (void**)&d_eta, sizeof(double) );
    cudaMalloc( (void**)&d_n, sizeof(double) );
    cudaMalloc( (void**)&d_beta, sizeof(double) );
    
    cudaMemset( d_n, 0, sizeof(double) );
    
    cudaMemcpy(d_eta, &eta, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_beta, &beta, sizeof(double), cudaMemcpyHostToDevice);
    
    // kernel
    calcInnerLoop<<<1,1>>>( d_n, h, d_eta, d_beta );


    
    
    print_GPU<<<1,1>>>(d_n);
    cudaDeviceSynchronize();
    
}