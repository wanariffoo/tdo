#ifndef ASSEMBLE_H
#define ASSEMBLE_H

#include <vector>

#include <string>
#include <fstream>
#include <stdexcept>
#include <sstream>

using namespace std;


 
class Assembler{

public:
    Assembler(size_t dim, double h, vector<size_t> N, double youngMod, double poisson, double rho, size_t p, size_t numLevels);

    bool init(double* &d_A_local, vector<double*> &d_value, vector<size_t*> &d_index, vector<double*> &d_p_value, vector<size_t*> &d_p_index, vector<double*> &d_r_value, vector<size_t*> &d_r_index, double* &d_chi, vector<size_t> &num_rows, vector<size_t> &max_row_size, vector<size_t> &p_max_row_size, vector<size_t> &r_max_row_size, vector<size_t*> &d_node_index);
    bool init_GPU(double* &d_A_local, vector<double*> &d_value, vector<size_t*> &d_index, vector<double*> &d_p_value, vector<size_t*> &d_p_index, vector<double*> &d_r_value, vector<size_t*> &d_r_index, double* &d_chi, vector<size_t> &num_rows, vector<size_t> &max_row_size, vector<size_t> &p_max_row_size, vector<size_t> &r_max_row_size, vector<size_t*> &d_node_index, ofstream& ofssbm);

    ~Assembler();

    bool assembleLocal();
    bool assembleLocal_(); // TODO: delete later
    bool assembleProlMatrix(size_t lev);
    bool assembleProlMatrix_GPU(vector<double*> &d_p_value, vector<size_t*> &d_p_index, size_t lev);
    bool assembleRestMatrix(size_t lev);
    bool assembleRestMatrix_GPU(vector<double*> &d_r_value, vector<size_t*> &d_r_index, vector<double*> &d_p_value, vector<size_t*> &d_p_index);
    bool assembleGlobal(vector<size_t> &num_rows, vector<size_t> &max_row_size, vector<size_t> &p_max_row_size, vector<size_t> &r_max_row_size);
    void setBC(vector<vector<size_t>> bc_index);
    bool UpdateGlobalStiffness(double* &d_chi, vector<double*> &d_value, vector<size_t*> &d_index, vector<double*> &d_p_value, vector<size_t*> &d_p_index, vector<double*> &d_r_value, vector<size_t*> &d_r_index, double* &d_A_local);
    
 
    vector<size_t> getNumNodesPerDim();
    vector<size_t> getGridSize();
    size_t getNumElements();
    size_t getNumNodes();

    double valueAt(size_t x, size_t y);

    class Node
    {
        public:
            Node (int id);
            int index();
    
        private:
            int m_index;
            float m_coo[2];
            int m_dof[2];
            // vector<int> m_dof(2);
    };

    class Element
    {
        public:
            Element(int ind);

            size_t index();
            void addNode(Node *x);
            int nodeIndex(int i);
            void printNodes();
            double valueAt(size_t x, size_t y, size_t num_cols);
            size_t getNodeIndex(int index);
            
        private:
            std::vector<Node*> m_node;
            size_t m_index;
        
    };


private:


    // grid dimensions
    vector<vector<size_t>> m_N;

    double m_h;
    size_t m_dim;
    size_t m_num_rows_l;    // local
    vector<size_t> m_num_rows;    // global stiffness matrix of each grid-level

    // material properties
    double m_youngMod;
    double m_poisson;
    vector<double> m_E;

    // stiffness matrices of each level
    vector<vector<vector<double>>> m_A_g;

    // prolongation matrices of each level
    vector<vector<vector<double>>> m_P;
    size_t m_p_num_rows;
    size_t m_p_num_cols;

    // restriction matrices of each level
    vector<vector<vector<double>>> m_R;
    size_t m_r_num_rows;
    size_t m_r_num_cols;

    // multi-grid
    size_t m_numLevels;
    size_t m_topLev;

    // boundary condition index
    vector<vector<size_t>> m_bc_index;
    vector<size_t*> m_d_bc_index;

    //// TDO
    double m_rho;
    size_t m_p;
    vector<double> m_chi;
    double del_t;
    
    //weighted average driving force
    double m_p_w; 

    // local stiffness matrix (dense)
    vector<double> m_A_local;

    // prolongation matrices on each level
    vector<vector<double>> m_p_value_g;
    vector<vector<size_t>> m_p_index_g;    
    vector<size_t> m_p_max_row_size;

    // restriction matrices on each level
    vector<vector<double>> m_r_value_g;
    vector<vector<size_t>> m_r_index_g;    
    vector<size_t> m_r_max_row_size;


    // global stiffness matrix's ELLPACK vectors on each grid-level
    vector<vector<double>> m_value_g;
    vector<vector<size_t>> m_index_g;
    vector<size_t> m_max_row_size; 

    //
    vector<size_t> m_numElements;
    vector<vector<size_t>> m_numNodesPerDim;
    vector<size_t> m_numNodes;

    
    vector<Node> m_node;
    vector<Element> m_element;

    vector<vector<size_t>> m_node_index;    

    //// CUDA

    // cuda block dimension for ellpack matrices
    vector<dim3> m_ell_gridDim;
    vector<dim3> m_ell_blockDim;

    // local stiffness matrix
    double* m_d_A_local;

    vector<size_t*> m_d_node_index;
    size_t* d_bc_index;

    double* m_d_temp_matrix;

    vector<cudaStream_t> m_streams;

    // rows and max_row_sizes
    size_t* m_d_num_rows;
    size_t* m_d_max_row_size;
    size_t* m_d_p_max_row_size;
    size_t* m_d_r_max_row_size;

   
};



#endif //ASSEMBLE_H