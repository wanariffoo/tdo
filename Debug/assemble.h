#ifndef ASSEMBLE_H
#define ASSEMBLE_H

#include <vector>

using namespace std;


 
class Assembler{

public:
    Assembler(size_t dim, double youngMod, double poisson);

    bool init();

    ~Assembler();

    bool set_domain_size(size_t h, size_t Nx, size_t Ny);

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
            Element(size_t ind);

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
    size_t m_Nx;
    size_t m_Ny;
    size_t m_Nz;

    size_t m_h;
    size_t m_dim;
    size_t m_num_rows_l;    // local
    size_t m_num_rows_g;    // global

    // material properties
    double m_youngMod;
    double m_poisson;
    vector<double> m_E;

    // stiffness matrices of each level
    vector<vector<double>> m_A_g;

    // prolongation matrices of each level
    vector<vector<double>> m_P;
    size_t m_p_num_rows;
    size_t m_p_num_cols;


    vector<double> m_A_local;
    vector<size_t> m_bc_index;

    vector<double> m_value_g;
    vector<size_t> m_index_g;


    // device pointers
    double* d_m_A_local;

    size_t m_max_row_size; // global

    //
    size_t m_numElements;
    vector<size_t> m_numNodesPerDim;
    size_t m_numNodes;

    
    vector<Node> m_node;
    vector<Element> m_element;
};



#endif //ASSEMBLE_H