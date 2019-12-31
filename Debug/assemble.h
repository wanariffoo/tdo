#ifndef ASSEMBLE_H
#define ASSEMBLE_H

#include <vector>

using namespace std;


 
class Assembler{

public:
    Assembler(size_t dim, size_t h, vector<size_t> N, double youngMod, double poisson, size_t numLevels);

    bool init();

    ~Assembler();

    bool assembleLocal();
    bool assembleGlobal();
    void setBC(vector<size_t> bc_index);

    vector<double> assembleLocal_(double youngMod, double poisson);
    // bool updateStiffMatrix();

    double valueAt(size_t x, size_t y);

    class Node
    {
        public:
            Node (int id);

            void setXCoor(float x);
            void setYCoor(float y);
            float getXCoor(float x);
            float getYCoor(float y);
            void printCoor();
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

    size_t m_h;
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



    // CUDA

    // local stiffness matrix
    // ELLPACK format is not used as it is a dense matrix
    double* d_m_A_local;

    // global stiffness matrix on each grid-level
    vector<double*> d_m_value_g;
    vector<size_t*> d_m_index_g;

    // prolongation matrices
    vector<double*> d_m_p_value_g;
    vector<size_t*> d_m_p_index_g;
    
    
};



#endif //ASSEMBLE_H