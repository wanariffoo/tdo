#ifdef CUDA
    #include "ell_matrix_cuda.h"
#else
    #include "ell_matrix_mpi.h"
#endif