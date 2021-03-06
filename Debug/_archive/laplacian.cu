/*
    
*/

#include <iostream>
#include <cuda.h>
#include <vector>
#include <cuda_runtime.h>
#include "../include/mycudaheader.h"

using namespace std;

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


int main()
{

    double rho = 0.4;
    double lambda_trial = 0;
    double lambda_min;
    double lambda_max;
    double del_t = 1;
    double eta = 12;
    double beta = 0.5;
    
    vector<double> array = { 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4 };

    double* h_array = &array[0];

    // cout << laplacian(h_array, 0, 3) << endl;
    // cout << laplacian(h_array, 1, 3) << endl;
    // cout << laplacian(h_array, 2, 3) << endl;
    // cout << laplacian(h_array, 3, 3) << endl;
    // cout << laplacian(h_array, 4, 3) << endl;
    // cout << laplacian(h_array, 5, 3) << endl;
    // cout << laplacian(h_array, 6, 3) << endl;
    // cout << laplacian(h_array, 7, 3) << endl;
    // cout << laplacian(h_array, 8, 3) << endl;

}