#ifndef SOLVER_H
#define SOLVER_H

#include <vector>
using namespace std;

class Solver
{
public:

    // constructor
    Solver(vector<double*> d_value, vector<size_t*> d_index, vector<size_t> max_row_size, vector<double*> d_p_value, vector<size_t*> d__pindex, vector<size_t> p_max_row_size,double* d_u, double* d_b, size_t numLevels, vector<size_t> num_rows, vector<size_t> num_cols);

    // deconstructor
    // TODO: deallocation of device variables
    ~Solver();

    bool init();

    bool solve(double* d_u, double* d_r);

    bool precond(double* d_c, double* d_r);

    bool precond_add_update_GPU(double* d_c, double* d_r, std::size_t lev, int cycle);
    
    void set_num_prepostsmooth(size_t pre_n, size_t post_n);

    bool smoother(double* d_c, double* d_r, int lev);

    void set_cycle(const char type);


private:

    vector<double*> m_d_value;
    vector<size_t*> m_d_index;
    vector<size_t> m_max_row_size;
    vector<double*> m_d_p_value;
    vector<size_t*> m_d_p_index;
    vector<size_t> m_p_max_row_size;
    vector<size_t> m_num_rows;
    vector<size_t> m_num_cols;
    size_t m_numLevels;
    size_t m_topLev;

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
    vector<dim3> m_gridDim_cols;
    vector<dim3> m_blockDim_cols;

    // gmg-cycle
    int m_gamma;



};



#endif // SOLVER_H