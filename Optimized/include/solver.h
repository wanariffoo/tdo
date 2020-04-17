#ifndef SOLVER_H
#define SOLVER_H

#include <vector>

#include <string>
#include <fstream>
#include <stdexcept>
#include <sstream>

using namespace std;

class Solver
{
public:

    // constructor
    // Solver(vector<double*> d_value, vector<size_t*> d_index, vector<double*> d_p_value, vector<size_t*> d_p_index, size_t numLevels, vector<size_t> num_rows, vector<size_t> max_row_size, vector<size_t> p_max_row_size, double damp);
    Solver(vector<double*> d_value, vector<size_t*> d_index, vector<size_t> max_row_size, vector<double*> d_p_value, vector<size_t*> d_p_index, vector<size_t> p_max_row_size, vector<double*> d_r_value, vector<size_t*> d_r_index, vector<size_t> r_max_row_size, size_t numLevels, vector<size_t> num_rows, double damp);

    ~Solver();

    bool init();

    bool solve(double* d_u, double* d_r, vector<double*> d_value, ofstream& ofssbm);
    // ofstream& ofssbm

    bool base_solve(double* d_bs_u, double* d_bs_b, ofstream& ofssbm);
    

    bool precond(double* d_c, double* d_r, ofstream& ofssbm);
    // bool precond_(double* d_c, double* d_r, double* d_value, size_t* d_index, size_t max_row_size, size_t m_num_rows);

    bool precond_add_update_GPU(double* d_c, double* d_r, std::size_t lev, int cycle, ofstream& ofssbm);
    
    void set_num_prepostsmooth(size_t pre_n, size_t post_n);
    void set_convergence_params(size_t maxIter, double minRes, double minRed);
    void set_bs_convergence_params(size_t maxIter, double minRes, double minRed);
    void set_convergence_params_( size_t maxIter, size_t bs_maxIter, double minRes, double minRed );

    bool smoother(double* d_c, double* d_r, int lev);

    // reinitialize the relevant device variables to zero, in order to solve with the updated stiffness matrix
    bool reinit();

    void set_cycle(const char type);
    void set_verbose(bool verbose, bool bs_verbose);

    // DEBUG:
    void set_steps(size_t step, size_t bs_step);


private:

    bool m_verbose;
    bool m_bs_verbose;

    vector<double*> m_d_value;
    vector<size_t*> m_d_index;
    vector<size_t> m_max_row_size;
    vector<double*> m_d_p_value;
    vector<size_t*> m_d_p_index;
    vector<size_t> m_p_max_row_size;
    vector<double*> m_d_r_value;
    vector<size_t*> m_d_r_index;
    vector<size_t> m_r_max_row_size;
    vector<size_t> m_num_rows;
    vector<size_t> m_num_cols;
    size_t m_numLevels;
    size_t m_topLev;

    size_t m_maxIter;
	double m_minRes;
	double m_minRed;

    // residuum vector
    double* m_d_r;

    // correction vector
    double* m_d_c;

    // TODO: change comment, not previous
    // previous residuum 
    double *m_d_res0;
    
    // last residuum
    double *m_d_lastRes;

    // current residuum
    double *m_d_res;

    // minimum required residuum for convergence
    double *m_d_minRes;
    
    // minimum required reduction for convergence
    double *m_d_minRed;

    // iteration steps
    size_t* m_d_step;     // solver
    size_t* m_d_bs_step;  // base solver
    
    // convergence checks
    bool m_foo = true;
    bool m_bs_foo = true;
    bool* m_d_foo;
    bool* m_d_bs_foo;

    // gmg's correction and residuum vectors of each level
    vector<double*> m_d_gmg_c;
    vector<double*> m_d_gmg_r;

    // temporary residuum vectors for GMG
    vector<double*> m_d_rtmp;

    // temporary correction vectors for GMG
    vector<double*> m_d_ctmp;

    // smoother damping parameter
    double m_damp;

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

    /////////////////
    // base solver //
    /////////////////

    size_t m_bs_maxIter;
    double m_bs_minRes;
    double m_bs_minRed;

    // bs residuum vector
    double* m_d_bs_r;

    // bs conjugate descent direction
    double* m_d_bs_p;

    // bs precond vector
    double* m_d_bs_z;

    double* m_d_bs_rho;
    double* m_d_bs_rho_old;

    double* m_d_bs_alpha;
    double* m_d_bs_alpha_temp;  // TODO: maybe use a general temp here

    // bs residuum variable
    double* m_d_bs_res;
    double* m_d_bs_res0;
    // double* m_d_bs_minRes;
    // double* m_d_bs_minRed;
    double* m_d_bs_lastRes;

    // DEBUG:
    size_t m_step;
    size_t m_bs_step;

        

};



#endif // SOLVER_H