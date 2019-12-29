#include <iostream>
#include "assemble.h"
#include <cmath>
#include "cudakernels.h"

using namespace std;


// Node class definitions
Assembler::Node::Node (int id) : m_index(id){}

void Assembler::Node::setXCoor(float x) { m_coo[0] = x; }
void Assembler::Node::setYCoor(float y) { m_coo[1] = y; }
float Assembler::Node::getXCoor(float x) { return m_coo[0]; }
float Assembler::Node::getYCoor(float y) { return m_coo[1]; }
int Assembler::Node::index() { return m_index; }


void Assembler::Node::printCoor()
{
    cout << "node [" << m_index << "] = ( " << m_coo[0] << ", " << m_coo[1] << " )" << endl;
}


// Element class definitions
Assembler::Element::Element(size_t ind) : m_index(ind)
{
    
}

double Assembler::Element::valueAt(size_t x, size_t y, size_t num_cols)
{
    // return m_A_local[ y + x * num_cols ];
    // cout << m_A_local << endl;

    return 0;
}

size_t Assembler::Element::index() { return m_index; }

void Assembler::Element::addNode(Node *x) { m_node.push_back(x); }

int Assembler::Element::nodeIndex(int i) { return m_node[i]->index(); }

void Assembler::Element::printNodes() 
{
    cout << "Element " << m_index << ": nodes = { ";
    for ( int i = 0 ; i < m_node.size() - 1 ; ++i )
        cout << m_node[i]->index() << ", ";

    cout << m_node[ m_node.size() - 1]->index() << " }" << endl;
}



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
    // CUDA_CALL( cudaFree(d_m_A_local) );
}

bool Assembler::set_domain_size(size_t h, size_t Nx, size_t Ny)
{
    m_h = h;
    m_Nx = Nx;
    m_Ny = Ny;
    return true;
}

bool Assembler::init()
{

    if ( m_dim == 2 )
    {
        m_A_local.resize(64, 0.0);
        m_num_rows_l = 8;
    }

    else if (m_dim == 3 )
    {
        m_A_local.resize(144, 0.0);
        m_num_rows_l = 12;
    }

    else
        cout << "error" << endl; //TODO: add error/assert
    
    // calculate the number of nodes/elements in the domain
    m_numElements = m_Nx * m_Ny;
    
    m_numNodesPerDim.resize(m_dim);

    // TODO: for loop, possibly change Nx, Ny, .. to vector<size_t> numElements
    m_numNodesPerDim[0] = m_Nx + 1;
    m_numNodesPerDim[1] = m_Ny + 1;

    m_numNodes = 1;
    for ( int i = 0 ; i < m_dim ; i++ )
        m_numNodes *= m_numNodesPerDim[i];


    // num of rows in global stiffness matrix
    m_num_rows_g = m_numNodes * m_dim;

    assembleLocal();
    assembleGlobal();
    // cout << m_A_local[0] << endl;

    // CUDA_CALL( cudaMalloc((void**)&d_m_A_local, sizeof(double) * m_A_local.size()) );
    // CUDA_CALL( cudaMemcpy(d_m_A_local, &m_A_local[0], sizeof(double) * m_A_local.size(), cudaMemcpyHostToDevice) );
    
    return true;

}

double Assembler::valueAt(size_t x, size_t y)
{
    return m_A_local[y + x*m_num_rows_l];
}

// to produce an ELLmatrix of the global stiffness in the device
// will return d_value, d_index, d_max_row_size
bool Assembler::assembleGlobal()
{
    // TODO: if no BC is set, return false with error

    for ( int i = 0 ; i < m_numNodes ; ++i )
    {
        m_node.push_back(Node(i));
    }

    for ( int i = 0 ; i < m_numElements ; ++i )
    {
        m_element.push_back(Element(i));
    }

    // assign the nodes to each element
    for ( int i = 0 ; i < m_numElements ; i++ )
    {
        m_element[i].addNode(&m_node[ i + i/m_Nx ]);   // lower left node
        m_element[i].addNode(&m_node[ i + i/m_Nx + 1]);   // lower right node
        m_element[i].addNode(&m_node[ i + i/m_Nx + m_Nx + 1]);   // upper left node
        m_element[i].addNode(&m_node[ i + i/m_Nx + m_Nx + 2]);   // upper right node
    }

    // cout << m_node[0].index() << endl;
    // m_element[0].printNodes();

    // create a function for this so that A_g is temporary
    
    // double A_g[m_numNodes*m_dim][m_numNodes*m_dim];
    vector<vector<double>> A_g (m_numNodes*m_dim, std::vector <double> (m_numNodes*m_dim, 0.0));

    for ( int i = 0 ; i < m_numNodes*m_dim; i++)
    {
        for ( int j = 0 ; j < m_numNodes*m_dim; j++)
            A_g[i][j] = 0;
    }

    cout << m_numElements << endl;
    

    // TODO: !!!
    see TDO sheet 12, check the discrepencies
    think it's due to this for loop below here:


    for ( int elmn_index = 0 ; elmn_index < m_numElements ; elmn_index++ )
    {
        for ( int x = 0 ; x < 4 ; x++ ) // TODO: dim  
        {
            for ( int y = 0 ; y < 4 ; y++ )        // TODO: dim   
            {      

                    A_g[ 2*m_element[elmn_index].nodeIndex(x)     ][ 2*m_element[elmn_index].nodeIndex(y)     ] += valueAt( 2*x    , 2*y );
                    A_g[ 2*m_element[elmn_index].nodeIndex(x)     ][ 2*m_element[elmn_index].nodeIndex(y) + 1 ] += valueAt( 2*x    , 2*y + 1      );
                    A_g[ 2*m_element[elmn_index].nodeIndex(x) + 1 ][ 2*m_element[elmn_index].nodeIndex(y)     ] += valueAt( 2*x + 1, 2*y          );
                    A_g[ 2*m_element[elmn_index].nodeIndex(x) + 1 ][ 2*m_element[elmn_index].nodeIndex(y) + 1 ] += valueAt( 2*x + 1, 2*y + 1      );
            }
        }
    }

    // for ( int i = 0 ; i < m_bc_index.size() ; ++i )
    //     applyMatrixBC(A_g, m_bc_index[i], m_num_rows_g);

    for ( int i = 0 ; i < m_num_rows_g ; i++ )
    {
        for ( int j = 0 ; j < m_num_rows_g ; j++ )
            cout << A_g[i][j] << " ";
        
        cout << "\n";
    }





    // calculate global max_num_rows
    m_max_row_size = getMaxRowSize(A_g, m_num_rows_g);


    // transformtoELL
    // transformToELL(A_g, m_value_g, m_index_g, m_max_row_size, m_num_rows_g);

    // for ( int i = 0 ; i < m_max_row_size * m_num_rows_g ; i++ )
    // cout << m_value_g[i] << " ";

    // cout << "\n";

    





    return true;

}


// TODO: check this is not right!
// assembles the local stiffness matrix
bool Assembler::assembleLocal()
{
    cout << "assembleLocal" << endl;

    // TODO: you haven't added JACOBI, see "TODO:" just before this function's return true

 

    double E[3][3];

    E[0][0] = E[1][1] = m_youngMod/(1 - m_poisson * m_poisson );
    E[0][1] = E[1][0] = m_poisson * E[0][0];
    E[2][2] = (1 - m_poisson) / 2 * E[0][0];
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
                m_A_local[j + i*m_num_rows_l] += bar[GP][i][j];     // TODO: * jacobi here
        }
    }


    return true;
}

void Assembler::setBC(vector<size_t> bc_index)
{
    m_bc_index = bc_index;
}

// assembles the local stiffness matrix
vector<double> Assembler::assembleLocal_(double youngMod, double poisson)
{
    cout << "assembleLocal" << endl;
    vector<double> A_local;
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


    return A_local;
}
