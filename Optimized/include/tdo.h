#ifndef TDO_H
#define TDO_H

#include <vector>

class TDO{

public:
    TDO(double* d_u, double* d_chi, double h, size_t dim, double beta, double eta, size_t numElements, size_t num_rows, double* d_A_local, vector<size_t*> d_node_index, vector<size_t> N, double rho, size_t numLevels, size_t p, vector<size_t> nodeindex);
    bool init();
    bool innerloop(double* &d_u, double* &d_chi);
    void set_verbose(bool verbose);
    void print_VTK(bool foo=false);
    

private:

    bool m_verbose;

    // grid
    size_t m_dim;
    double m_h;
    double m_local_volume;
    size_t m_numElements;
    size_t m_num_rows;
    size_t m_Nx;
    size_t m_Ny;
    size_t m_Nz;
    size_t m_numLevels;

    // inner loop
    size_t m_n;
    size_t m_j;

    // TDO
    double m_del_t;
    double m_rho;
    size_t m_p;

    // VTK
    int m_file_index = 0;
    bool m_printVTK = false;

    double m_betastar;
    double m_etastar;

    double* m_d_chi;

    dim3 m_gridDim;
    dim3 m_blockDim;

    // CUDA

    // displacement
    double* m_d_u;

    // local stiffness matrix
    double* m_d_A_local;

    // driving force of each element
    double* m_d_df;

    // weighted average driving force
    double* m_d_p_w;

    int *m_d_mutex;

    double* m_d_beta;
    double* m_d_eta;

    double *m_d_temp;
    double *m_d_temp_s; // scalar

    double *m_d_uTAu;
    vector<size_t*> m_d_node_index;

    // bisection algorithm
    double* m_d_lambda_l;
    double* m_d_lambda_u;
    double* m_d_lambda_tr;
    
    double* m_d_chi_tr;

    //NOTE: reuse this from somewhere? temp variable?
    double* m_d_rho_tr;   

    // convergence check
    bool m_tdo_foo = true;
    bool* m_d_tdo_foo;

    // DEBUG: temporary
    vector<double> laplacian;
    double* d_laplacian;
    
    // DEBUG: test
    vector<size_t> m_node_index;
    size_t* m_d_node_index_;

    double* m_d_sum_df_g;
    double* m_d_sum_g;


    

};

#endif // TDO_H