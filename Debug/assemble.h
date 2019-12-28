#ifndef ASSEMBLE_H
#define ASSEMBLE_H

#include <vector>

using namespace std;

class Assembler{

public:
    Assembler(size_t dim, double youngMod, double poisson);
    bool assembleLocal();
    bool assembleMaterialMat();
    // bool updateStiffMatrix();


private:
    // grid dimensions
    size_t m_Nx;
    size_t m_Ny;
    size_t m_Nz;

    size_t m_dim;

    // material matrix
    vector<double> m_E;

    vector<double> m_A_local;

    // device pointers
    double* d_m_A_local;

    

};



#endif //ASSEMBLE_H