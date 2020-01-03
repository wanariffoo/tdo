#ifndef TDO_H
#define TDO_H

#include <vector>

class TDO{

public:
    TDO(double* d_u, double* d_kai, double h, size_t dim, double beta, double eta, size_t numElements, double* d_A_local);
    bool init();
    bool innerloop();



private:

    // grid
    size_t m_dim;
    double m_h;
    double m_local_volume;
    size_t m_numElements;

    // inner loop
    size_t m_n;
    size_t m_j;


    double m_beta;
    double m_eta;

    double* m_d_kai;

    dim3 m_gridDim;
    dim3 m_blockDim;

    // CUDA

    // displacement
    double* m_d_u;

    // local stiffness matrix
    double* m_d_A_local;

    // driving force of each element
    vector<double*> m_d_df;
    double* m_d_uTAu;

    int *m_d_mutex;

    double *m_d_temp;
    vector<size_t*> m_d_node_index;




};

#endif // TDO_H