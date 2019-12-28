#include <iostream>
#include "assemble.h"
#include "cudakernels.h"

using namespace std;

Assembler::Assembler(size_t dim, double youngMod, double poisson)
{
    cout << "assembler" << endl;

    m_youngMod = youngMod;
    m_poisson = poisson;

    m_dim = dim;

}

Assembler::~Assembler()
{
    cout << "assembler : deallocate" << endl;
    CUDA_CALL( cudaFree(d_m_A_local) );
}

bool Assembler::set_domain_size(size_t h, size_t Nx, size_t Ny)
{
    return true;
}

bool Assembler::init()
{
    assembleLocal(m_A_local, m_youngMod, m_poisson);


    CUDA_CALL( cudaMalloc((void**)&d_m_A_local, sizeof(double) * m_A_local.size()) );
    CUDA_CALL( cudaMemcpy(d_m_A_local, &m_A_local[0], sizeof(double) * m_A_local.size(), cudaMemcpyHostToDevice) );
    

    return true;

}

bool Assembler::assembleGlobal()
{
    // TODO: if no BC is set, return false with error


    return true;
}

// assembles the local stiffness matrix
bool Assembler::assembleLocal(vector<double> &A_local, double youngMod, double poisson)
{
    cout << "assembleLocal" << endl;

    // TODO: you haven't added JACOBI, see "TODO:" just before this function's return true

    size_t num_cols;

    if ( m_dim == 2 )
    {
        A_local.resize(64, 0.0);
        num_cols = 8;
    }

    else if (m_dim == 3 )
    {
        A_local.resize(144, 0.0);
        num_cols = 12;
    }

    else
        cout << "error" << endl; //TODO: add error/assert

    double E[3][3];

    E[0][0] = E[1][1] = youngMod/(1 - poisson * poisson );
    E[0][1] = E[1][0] = poisson * E[0][0];
    E[2][2] = (1 - poisson) / 2 * E[0][0];
    E[2][0] = E[2][1] = E[1][2] = E[0][2];

    // bilinear shape function matrix (using 4 Gauss Points)
    double B[4][3][8] = { { {-0.3943375,	0,	0.3943375,	0,	0.1056625,	0,	-0.1056625,	0}, {0,	-0.3943375,	0,	-0.1056625,	0,	0.1056625,	0,	0.3943375} , {-0.3943375,	-0.3943375,	-0.1056625,	0.3943375,	0.1056625,	0.1056625,	0.3943375,	-0.1056625} },
                          { {-0.3943375,	0,	0.3943375,	0,	0.1056625,	0,	-0.1056625,	0}, {0,	-0.1056625,	0,	-0.3943375,	0,	0.3943375,	0,	0.1056625}, {-0.1056625,	-0.3943375,	-0.3943375,	0.3943375,	0.3943375,	0.1056625,	0.1056625,	-0.1056625} },
                          { {-0.1056625,	0,	0.1056625,	0,	0.3943375,	0,	-0.3943375,	0}, {0,	-0.3943375,	0,	-0.1056625,	0,	0.1056625,	0,	0.3943375}, {-0.3943375,	-0.1056625,	-0.1056625,	0.1056625,	0.1056625,	0.3943375,	0.3943375,	-0.3943375} },
                          { {-0.1056625,	0,	0.1056625,	0,	0.3943375,	0,	-0.3943375,	0}, {0,	-0.1056625,	0,	-0.3943375,	0,	0.3943375,	0,	0.1056625}, {-0.1056625,	-0.1056625,	-0.3943375,	0.1056625,	0.3943375,	0.3943375,	0.1056625,	-0.3943375} }
                        };

    // 4 matrices with size 3x8 to store each GP's stiffness matrix
    double foo[4][3][8];
    double bar[4][8][8];

    // intializing to zero
    for ( int GP = 0 ; GP < 4 ; GP++)
    {
        for ( int i = 0 ; i < 8 ; i++ )
        {
            for( int j = 0 ; j < 3 ; j++ )
                foo[GP][j][i] = 0;

            for( int j = 0 ; j < 8 ; j++ )
                bar[GP][j][i] = 0;
        }
    }

    // calculating A_local = B^T * E * B

    // foo = E * B
    for ( int GP = 0 ; GP < 4 ; GP++)
    {
        for ( int i = 0 ; i < 3 ; i++ )
        {
            for( int j = 0 ; j < 8 ; j++ )
            {
                for ( int k = 0 ; k < 3 ; k++)
                    foo[GP][i][j] += E[i][k] * B[GP][k][j];
            }
        }
    }

    
    // bar = B^T * foo
    for ( int GP = 0 ; GP < 4 ; GP++)
    {
        for ( int i = 0 ; i < 8 ; i++ )
        {
            for( int j = 0 ; j < 8 ; j++ )
            {
                for ( int k = 0 ; k < 3 ; k++)
                    bar[GP][i][j] += B[GP][k][i] * foo[GP][k][j];
            }
        }
    }


    for ( int GP = 0 ; GP < 4 ; GP++)
    {
        for ( int i = 0 ; i < 8 ; i++ )
        {
            for( int j = 0 ; j < 8 ; j++ )
                m_A_local[j + i*num_cols] += bar[GP][i][j];     // TODO: * jacobi here
        }
    }


    return true;
}
