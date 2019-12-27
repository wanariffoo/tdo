#ifndef SOLVER_H
#define SOLVER_H

#include <vector>
using namespace std;

class Solver
{
public:

    // constructor
    Solver(double* d_value, size_t* d_index, size_t max_row_size, double* d_u, double* d_b, size_t numLevels, size_t num_rows, size_t num_cols);

    // deconstructor
    // TODO: deallocation of device variables
    ~Solver();

    bool init();

    bool solve(double* d_u, double* d_r);

    bool precond(double* d_c, double* d_r);
    
    void set_num_presmooth(size_t n);
    void set_num_postsmooth(size_t n);


private:

    double* m_d_value;
    size_t* m_d_index;
    size_t m_max_row_size;
    size_t m_num_rows;
    size_t m_num_cols;
    size_t m_numLevels;

    // residuum vector
    double* m_d_r;

    // correction vector
    double* m_d_c;

    // previous residuum
    double *m_d_res0;
    
    // current residuum
    double *m_d_res;

    // minimum required residuum for convergence
    double *m_d_m_minRes;
    
    // minimum required reduction for convergence
    double *m_d_m_minRed;
    
    // gmg's correction and residuum vectors of each level
    vector<double*> m_d_gmg_c;
    vector<double*> m_d_gmg_r;

    // temporary residuum vectors for GMG
    vector<double*> m_d_rtmp;

    // temporary correction vectors for GMG
    vector<double*> m_d_ctmp;

    // number of pre-/post-smooth cycles
    size_t m_numPreSmooth;
    size_t m_numPostSmooth;

    // cuda grid and block size for each level
    vector<dim3> m_gridDim;
    vector<dim3> m_blockDim;



};



#endif // SOLVER_H