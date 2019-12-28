#include <iostream>
#include "assemble.h"

using namespace std;

Assembler::Assembler(size_t dim, double youngMod, double poisson)
{
    cout << "assembler" << endl;

    m_dim = dim;
    double E[3][3];

    E[0][0] = E[1][1] = youngMod/(1 - poisson * poisson );
    E[0][1] = E[1][0] = poisson * E[0][0];
    E[2][2] = (1 - poisson) / 2 * E[0][0];
    E[2][0] = E[2][1] = E[1][2] = E[0][2];

    // assembling the material matrix

    // if ( m_dim == 2 ) 
    // {
    //     m_E.resize(9);
    //     m_E[0] = youngMod/(1 - poisson * poisson );
    //     m_E[4] = m_E[0];
    //     m_E[1] = poisson * m_E[0];
    //     m_E[3] = m_E[1];
    //     m_E[8] = (1 - poisson) / 2 * m_E[0];
        
    // }

    // bilinear shape function matrix
    // vector<vector<double>> N_ = {
    //     { -0.3943375,	0.3943375,	0.1056625,	-0.1056625,	-0.3943375,	-0.1056625,	0.1056625,	0.3943375},    // GP-1
    //     { -0.3943375,	0.3943375,	0.1056625,	-0.1056625,	-0.1056625,	-0.3943375,	0.3943375,	0.1056625},    // GP-2
    //     { -0.1056625,	0.1056625,	0.3943375,	-0.3943375,	-0.3943375,	-0.1056625,	0.1056625,	0.3943375},    // GP-3
    //     { -0.1056625,	0.1056625,	0.3943375,	-0.3943375,	-0.1056625,	-0.3943375,	0.3943375,	0.1056625}     // GP-4
    // };

    // double N[3][8] = {
    //     { -0.3943375,	0.3943375,	0.1056625,	-0.1056625,	-0.3943375,	-0.1056625,	0.1056625,	0.3943375},    // GP-1
    //     { -0.3943375,	0.3943375,	0.1056625,	-0.1056625,	-0.1056625,	-0.3943375,	0.3943375,	0.1056625},    // GP-2
    //     { -0.1056625,	0.1056625,	0.3943375,	-0.3943375,	-0.3943375,	-0.1056625,	0.1056625,	0.3943375},    // GP-3
    //     { -0.1056625,	0.1056625,	0.3943375,	-0.3943375,	-0.1056625,	-0.3943375,	0.3943375,	0.1056625}     // GP-4
    // };

    // matrix multiplication

    // double B[24] = {-0.3943375,	0,	0.3943375,	0,	0.1056625,	0,	-0.1056625,	0,
    //                 0,	-0.3943375,	0,	-0.1056625,	0,	0.1056625,	0,	0.3943375,
    //                 -0.3943375,	-0.3943375,	-0.1056625,	0.3943375,	0.1056625,	0.1056625,	0.3943375,	-0.1056625};

    double B[4][3][8] = { { {-0.3943375,	0,	0.3943375,	0,	0.1056625,	0,	-0.1056625,	0}, {0,	-0.3943375,	0,	-0.1056625,	0,	0.1056625,	0,	0.3943375} , {-0.3943375,	-0.3943375,	-0.1056625,	0.3943375,	0.1056625,	0.1056625,	0.3943375,	-0.1056625} },
                          { {-0.3943375,	0,	0.3943375,	0,	0.1056625,	0,	-0.1056625,	0}, {0,	-0.1056625,	0,	-0.3943375,	0,	0.3943375,	0,	0.1056625}, {-0.1056625,	-0.3943375,	-0.3943375,	0.3943375,	0.3943375,	0.1056625,	0.1056625,	-0.1056625} },
                          { {-0.1056625,	0,	0.1056625,	0,	0.3943375,	0,	-0.3943375,	0}, {0,	-0.3943375,	0,	-0.1056625,	0,	0.1056625,	0,	0.3943375}, {-0.3943375,	-0.1056625,	-0.1056625,	0.1056625,	0.1056625,	0.3943375,	0.3943375,	-0.3943375} },
                          { {-0.1056625,	0,	0.1056625,	0,	0.3943375,	0,	-0.3943375,	0}, {0,	-0.1056625,	0,	-0.3943375,	0,	0.3943375,	0,	0.1056625}, {-0.1056625,	-0.1056625,	-0.3943375,	0.1056625,	0.3943375,	0.3943375,	0.1056625,	-0.3943375} }
                        };

    // 4 matrices with size 3x8 to store each GP's stiffness matrix
    double foo[4][3][8];
    double bar[4][8][8];
    double A_l[8][8];

    // intializing to zero
    for ( int GP = 0 ; GP < 4 ; GP++)
    {
        for ( int i = 0 ; i < 3 ; i++ )
        {
            for( int j = 0 ; j < 8 ; j++ )
            {
                foo[GP][i][j] = 0.0;
                bar[GP][i][j] = 0.0;
                A_l[i][j] = 0.0;
            }
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

    // result = B^T * product
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
                A_l[i][j] += bar[GP][i][j];
        }
    }




}

// assembles the local stiffness matrix
bool Assembler::assembleLocal()
{
    cout << "assembleLocal" << endl;

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            


    return true;
}

bool Assembler::assembleMaterialMat()
{

    return true;
}