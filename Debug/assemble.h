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

    bool assembleLocal(vector<double> &A_local, double youngMod, double poisson);
    bool assembleGlobal()
    // bool updateStiffMatrix();


private:
    // grid dimensions
    size_t m_Nx;
    size_t m_Ny;
    size_t m_Nz;

    size_t m_dim;

    // material properties
    double m_youngMod;
    double m_poisson;
    vector<double> m_E;

    vector<double> m_A_local;

    // device pointers
    double* d_m_A_local;

    

};



#endif //ASSEMBLE_H