#ifndef ASSEMBLE_H
#define ASSEMBLE_H

#include <vector>

using namespace std;


 
class Assembler{

public:
    Assembler(size_t dim, double h, vector<size_t> N, double youngMod, double poisson, double rho, size_t p, size_t numLevels);

    bool init();

    ~Assembler();

    bool assembleLocal();
    bool assembleGlobal();
    void setBC(vector<size_t> bc_index);
 
    

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

    // multi-grid
    size_t m_numLevels;
    size_t m_topLev;

    vector<size_t> m_bc_index;

    ////// TDO
    double m_rho;
    size_t m_p;
    vector<double> m_kai;

    // local stiffness matrix (dense)
    vector<double> m_A_local;

    // prolongation matrices on each level
    vector<vector<double>> m_p_value_g;
    vector<vector<size_t>> m_p_index_g;    
    vector<size_t> m_p_max_row_size;


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



    ////// CUDA

    // local stiffness matrix
    // ELLPACK format is not used as it is a dense matrix
    double* d_A_l;

    // global stiffness matrix on each grid-level
    vector<double*> d_value;
    vector<size_t*> d_index;

    // prolongation matrices
    vector<double*> d_p_value;
    vector<size_t*> d_p_index;
    
    
};



#endif //ASSEMBLE_H