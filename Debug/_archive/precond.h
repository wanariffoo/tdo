#ifndef PRECOND_H
#define PRECOND_H

__global__
void Jacobi_Precond_GPU(double* c, double* value, double* r, size_t num_rows);

#endif // PRECOND_H