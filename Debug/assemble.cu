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
Assembler::Element::Element(int ind) : m_index(ind){}

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



Assembler::Assembler(size_t dim, size_t h, vector<size_t> N, double youngMod, double poisson, size_t numLevels)
    : m_dim(dim), m_h(h), m_youngMod(youngMod), m_poisson(poisson), m_numLevels(numLevels)
{
    cout << "assembler" << endl;

    

    // m_N [lev][dim]
    // e.g., m_N[lev=1][dim=0] = number of elements in x-dimension on grid-level 1
    m_N.resize(m_numLevels, vector<size_t>(m_dim));

    // storing the grid dimensions of each level
    for ( int lev = 0 ; lev < m_numLevels ; lev++ )
    {
        for ( int i = 0 ; i < m_dim ; i++ )
        {
            m_N[lev][i] = N[i];
            N[i] *= 2;
        }
    }

  


}

Assembler::~Assembler()
{
    cout << "assembler : deallocate" << endl;
    // CUDA_CALL( cudaFree(d_m_A_local) );
}


void Assembler::setBC(vector<size_t> bc_index)
{
    m_bc_index = bc_index;
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


    m_topLev = m_numLevels - 1;


    // TODO: perhaps combine these for loops into one? would it work?    
    // number of elements in each grid-level
    m_numElements.resize(m_numLevels, 1);
    
    for ( int lev = 0 ; lev < m_numLevels ; lev++ )
    {
        for ( int i = 0 ; i < m_dim ; i++)
            m_numElements[lev] *= m_N[lev][i];
    }


    // m_numNodesPerDim[lev][dim]
    m_numNodesPerDim.resize(m_numLevels, vector<size_t>(m_dim));

    for ( int lev = 0 ; lev < m_numLevels ; lev++ )
    {
        for ( int i = 0 ; i < m_dim ; i++)
            m_numNodesPerDim[lev][i] = m_N[lev][i] + 1;
    }

    // number of nodes per grid-level
    m_numNodes.resize(m_numLevels, 1);
    // m_numNodes = 1;

    for ( int lev = 0 ; lev < m_numLevels ; lev++ )
    {
        for ( int i = 0 ; i < m_dim ; i++ )
            m_numNodes[lev] *= m_numNodesPerDim[lev][i];
    }


    // num of rows in global stiffness matrix
    m_num_rows.resize(m_numLevels);

    for ( int lev = 0 ; lev < m_numLevels ; lev++ )
        m_num_rows[lev] = m_numNodes[lev] * m_dim;


    assembleLocal();

    assembleGlobal();

    // // CUDA_CALL( cudaMalloc((void**)&d_m_A_local, sizeof(double) * m_A_local.size()) );
    // // CUDA_CALL( cudaMemcpy(d_m_A_local, &m_A_local[0], sizeof(double) * m_A_local.size(), cudaMemcpyHostToDevice) );
    
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

    // TODO: use std::vector!!!
    // intializing to zero
    for ( int GP = 0 ; GP < 4 ; GP++)
    {
        for ( int i = 0 ; i < 8 ; i++ )
        {
            for( int j = 0 ; j < 3 ; j++ )
                foo[GP][j][i] = 0;

        }
    }

    /// calculating A_local = B^T * E * B
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
                    m_A_local[j + i*m_num_rows_l] += B[GP][k][i] * foo[GP][k][j];
            }
        }
    }
    
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

    // adding nodes and elements to the top-level global grid
    for ( int i = 0 ; i < m_numNodes[m_topLev] ; ++i )
        m_node.push_back(Node(i));

    for ( int i = 0 ; i < m_numElements[m_topLev] ; ++i )
        m_element.push_back(Element(i));


    // assign the nodes to each element
    for ( int i = 0 ; i < m_numElements[m_topLev] ; i++ )
    {
        m_element[i].addNode(&m_node[ i + i/m_N[m_topLev][0] ]);   // lower left node
        m_element[i].addNode(&m_node[ i + i/m_N[m_topLev][0] + 1]);   // lower right node
        m_element[i].addNode(&m_node[ i + i/m_N[m_topLev][0] + m_N[m_topLev][0] + 1]);   // upper left node
        m_element[i].addNode(&m_node[ i + i/m_N[m_topLev][0] + m_N[m_topLev][0] + 2]);   // upper right node
    }

    
    // // double A_g[m_numNodes*m_dim][m_numNodes*m_dim];
    // // TODO: figure out if you keep this as member var or not
    // m_A_g.resize(m_numNodes[m_topLev]*m_dim, vector<double>(m_numNodes[m_topLev]*m_dim));

    // resizing the global stiffness matrices on each grid-level
    m_A_g.resize(m_numLevels);

    for ( int lev = 0 ; lev < m_numLevels ; lev++ )
    {
        // number of columns in each level
        m_A_g[lev].resize(m_num_rows[lev]);
        
        // number of rows in each level
        for ( int j = 0 ; j < m_num_rows[lev] ; j++ )
                m_A_g[lev][j].resize(m_num_rows[lev]);
    }

    // filling in the global stiffness matrix from the local stiffness matrices of the 4 Gauss-Points
    for ( int elmn_index = 0 ; elmn_index < 4 ; elmn_index++ )
    {
        for ( int x = 0 ; x < 4 ; x++ ) // TODO: dim  
        {
            for ( int y = 0 ; y < 4 ; y++ )        // TODO: dim   
            {      

                    m_A_g[m_topLev][ 2*m_element[elmn_index].nodeIndex(x)     ][ 2*m_element[elmn_index].nodeIndex(y)     ] += valueAt( 2*x    , 2*y );
                    m_A_g[m_topLev][ 2*m_element[elmn_index].nodeIndex(x)     ][ 2*m_element[elmn_index].nodeIndex(y) + 1 ] += valueAt( 2*x    , 2*y + 1      );
                    m_A_g[m_topLev][ 2*m_element[elmn_index].nodeIndex(x) + 1 ][ 2*m_element[elmn_index].nodeIndex(y)     ] += valueAt( 2*x + 1, 2*y          );
                    m_A_g[m_topLev][ 2*m_element[elmn_index].nodeIndex(x) + 1 ][ 2*m_element[elmn_index].nodeIndex(y) + 1 ] += valueAt( 2*x + 1, 2*y + 1      );
            }
        }
    }


    
    // TODO: make a function for this
    // replacing any values <1e-7 to 0.0
    for ( int x = 0 ; x < m_numNodes[m_topLev]*m_dim ; x++ ) // TODO: dim  
    {
        for ( int y = 0 ; y < m_numNodes[m_topLev]*m_dim ; y++ )        // TODO: dim   
        {      
            if ( m_A_g[m_topLev][x][y] < 1e-7 && m_A_g[m_topLev][x][y] > -1e-7)
                m_A_g[m_topLev][x][y] = 0.0;
        }
    }

    //     // cout << m_A_g[15][14] << endl;

    // applying BC on the matrix
    // DOFs which are affected by BC will have identity rows { 0 0 .. 1 .. 0 0}
    for ( int i = 0 ; i < m_bc_index.size() ; ++i )
        applyMatrixBC(m_A_g[m_topLev], m_bc_index[i], m_num_rows[m_topLev]);


    /// resizing the prolongation matrices according to the number of grid-levels
    m_P.resize( m_numLevels - 1 );

    // CHECK: not sure!
    for ( int lev = 0 ; lev < m_numLevels - 1 ; lev++)
    {
        // number of columns in each level
        m_P[lev].resize(m_num_rows[lev]);
        
        // number of rows in each level
        for ( int j = 0 ; j < m_num_rows[lev] ; j++ )
                m_P[lev][j].resize(m_num_rows[lev + 1]);
    }


    // TODO: create a function to assemble the prolongation matrices in each level
    // assembleProlMatrices( m_P, m_numLevels )


    // DEBUG: temporary prolong
    m_P[0] =   {
            {1,	0,	0,	0,	0,	0,	0,	0},
            {0,	1,	0,	0,	0,	0,	0,	0},
            {0,	0,	0.5,	0,	0,	0,	0,	0},
            {0,	0,	0,	0.5,	0,	0,	0,	0},
            {0,	0,	1,	0,	0,	0,	0,	0},
            {0,	0,	0,	1,	0,	0,	0,	0},
            {0,	0,	0,	0,	0,	0,	0,	0},
            {0,	0,	0,	0,	0,	0,	0,	0},
            {0,	0,	0.25,	0,	0,	0,	0.25,	0},
            {0,	0,	0,	0.25,	0,	0,	0,	0.25},
            {0,	0,	0.5,	0,	0,	0,	0.5,	0},
            {0,	0,	0,	0.5,	0,	0,	0,	0.5},
            {0,	0,	0,	0,	1,	0,	0,	0},
            {0,	0,	0,	0,	0,	1,	0,	0},
            {0,	0,	0,	0,	0,	0,	0.5,	0},
            {0,	0,	0,	0,	0,	0,	0,	0.5},
            {0,	0,	0,	0,	0,	0,	1,	0},
            {0,	0,	0,	0,	0,	0,	0,	1}
            };


    // for ( int i = 0 ; i < m_num_rows[1] ; i++ )
    // {
    //     for ( int j = 0 ; j < m_num_rows[0] ; j++ )
    //         cout << m_P[0][i][j] << " ";

    //     cout << "\n";
    // }


    // TODO: transform later, after you get the A's in all levels
    



    // for ( int i = 0 ; i < m_num_rows[1] ; i++ )
    // {
    //     for ( int j = 0 ; j < m_num_rows[1] ; j++ )
    //         cout << m_A_g[1][i][j] << " ";

    //     cout << "\n";
    // }

    // // DEBUG: temp
    // std::vector<std::vector<double>> A_coarse ( m_num_rows_l, std::vector <double> (m_num_rows_g, 0.0));

    // // for ( int i = 0 ; i < 4 ; i++ )
    // // {
    // //     for ( int j = 0 ; j < 4 ; j++ )
    // //         cout << A_coarse[i][j] << " ";

    // //     cout << "\n";
    // // }

    // resizing the coarse stiffness matrices on each grid-level

    for ( int lev = 0 ; lev < m_numLevels - 1; lev++ )
        PTAP(m_A_g[lev], m_A_g[lev+1], m_P[lev], m_num_rows[lev+1], m_num_rows[lev] );

    // for ( int i = 0 ; i < m_num_rows[0] ; i++ )
    // {
    //     for ( int j = 0 ; j < m_num_rows[0] ; j++ )
    //         cout << m_A_g[0][i][j] << " ";

    //     cout << "\n";
    // }


    // TODO:
    // create global element ... Element global(-1) : if ind = -1, this is global
    // in this element, store all the node indices
    // have a function to find the N S E W nodes
    
    // Node getNeighbourNode(Node node, enum NSEW)

    // size_t node_index = getNeighbourNode()

    // A[ test.index ][ ] or something like that

    // use this to get the Node, and then 



    // // TODO: maybe do this together with the lower levels?
    // // calculate global max_num_rows, which will also be needed when allocating memory in device
    m_max_row_size.resize(m_numLevels);
    for ( int lev = 0 ; lev < m_numLevels ; lev++ )
        m_max_row_size[lev] = getMaxRowSize(m_A_g[lev], m_num_rows[lev], m_num_rows[lev]);

    m_p_max_row_size.resize ( m_numLevels - 1 );
    for ( int lev = 0 ; lev < m_numLevels - 1 ; lev++ )
        m_p_max_row_size[lev] = getMaxRowSize(m_P[lev], m_num_rows[lev+1], m_num_rows[lev]);
    
    cout << m_max_row_size[0] << endl;
    cout << m_p_max_row_size[0] << endl;
    // obtaining the ELLPACK value and index vectors from the global stiffness matrix
    
    m_p_value_g.resize( m_numLevels - 1 );
    m_p_index_g.resize( m_numLevels - 1 );
    // prolongation matrices
    for ( int lev = 0 ; lev < m_numLevels - 1 ; lev++ )
    {
        transformToELL(m_P[lev], m_p_value_g[lev], m_p_index_g[lev], m_p_max_row_size[lev], m_num_rows[lev+1]);    
    }


    // transformToELL(m_A_g, m_value_g, m_index_g, m_max_row_size, m_num_rows_g);

    for ( int i = 0 ; i < 36 ; i++ )
    {
        cout << m_p_value_g[0][i] << " ";
    }
        cout << "\n";

    // for ( int i = 0 ; i < m_num_rows[1] ; i++ )
    // {
    //     for ( int j = 0 ; j < m_num_rows[0] ; j++ )
    //         cout << m_P[0][i][j] << " ";

    //     cout << "\n";
    // }
    


    // NOTE: can somehow do init for solving now while allocating memory in device?
    // do async malloc then your init() should be AFTER the memcpy stuff, not before


    // CUDA
    // allocating memory in device

    // // local stiffness
    // CUDA_CALL( cudaMalloc((void**)&d_m_A_local, sizeof(double) * m_A_local.size() ) );

    // // global matrices on each grid-level
    // for ( int lev = 0 ; lev < m_numLevels ; lev++ )
    // {
    //     CUDA_CALL( cudaMalloc((void**)&m_value_g[lev], sizeof(double) * m_max_row_size[lev] * m_num_rows[lev] ) );
    //     CUDA_CALL( cudaMalloc((void**)&m_index_g[lev], sizeof(size_t) * m_max_row_size[lev] * m_num_rows[lev] ) );
    // }


    return true;

}



// // assembles the local stiffness matrix
// vector<double> Assembler::assembleLocal_(double youngMod, double poisson)
// {
    // cout << "assembleLocal" << endl;
    // vector<double> A_local;
    // // TODO: you haven't added JACOBI, see "TODO:" just before this function's return true

    // size_t num_cols;

    // if ( m_dim == 2 )
    // {
    //     A_local.resize(64, 0.0);
    //     num_cols = 8;
    // }

    // else if (m_dim == 3 )
    // {
    //     A_local.resize(144, 0.0);
    //     num_cols = 12;
    // }

    // else
    //     cout << "error" << endl; //TODO: add error/assert

    // double E[3][3];

    // E[0][0] = E[1][1] = youngMod/(1 - poisson * poisson );
    // E[0][1] = E[1][0] = poisson * E[0][0];
    // E[2][2] = (1 - poisson) / 2 * E[0][0];
    // E[2][0] = E[2][1] = E[1][2] = E[0][2];

    // // bilinear shape function matrix (using 4 Gauss Points)
    // double B[4][3][8] = { { {-0.3943375,	0,	0.3943375,	0,	0.1056625,	0,	-0.1056625,	0}, {0,	-0.3943375,	0,	-0.1056625,	0,	0.1056625,	0,	0.3943375} , {-0.3943375,	-0.3943375,	-0.1056625,	0.3943375,	0.1056625,	0.1056625,	0.3943375,	-0.1056625} },
    //                       { {-0.3943375,	0,	0.3943375,	0,	0.1056625,	0,	-0.1056625,	0}, {0,	-0.1056625,	0,	-0.3943375,	0,	0.3943375,	0,	0.1056625}, {-0.1056625,	-0.3943375,	-0.3943375,	0.3943375,	0.3943375,	0.1056625,	0.1056625,	-0.1056625} },
    //                       { {-0.1056625,	0,	0.1056625,	0,	0.3943375,	0,	-0.3943375,	0}, {0,	-0.3943375,	0,	-0.1056625,	0,	0.1056625,	0,	0.3943375}, {-0.3943375,	-0.1056625,	-0.1056625,	0.1056625,	0.1056625,	0.3943375,	0.3943375,	-0.3943375} },
    //                       { {-0.1056625,	0,	0.1056625,	0,	0.3943375,	0,	-0.3943375,	0}, {0,	-0.1056625,	0,	-0.3943375,	0,	0.3943375,	0,	0.1056625}, {-0.1056625,	-0.1056625,	-0.3943375,	0.1056625,	0.3943375,	0.3943375,	0.1056625,	-0.3943375} }
    //                     };

    // // 4 matrices with size 3x8 to store each GP's stiffness matrix
    // double foo[4][3][8];
    // double bar[4][8][8];

    // // intializing to zero
    // for ( int GP = 0 ; GP < 4 ; GP++)
    // {
    //     for ( int i = 0 ; i < 8 ; i++ )
    //     {
    //         for( int j = 0 ; j < 3 ; j++ )
    //             foo[GP][j][i] = 0;

    //         for( int j = 0 ; j < 8 ; j++ )
    //             bar[GP][j][i] = 0;
    //     }
    // }

    // // calculating A_local = B^T * E * B

    // // foo = E * B
    // for ( int GP = 0 ; GP < 4 ; GP++)
    // {
    //     for ( int i = 0 ; i < 3 ; i++ )
    //     {
    //         for( int j = 0 ; j < 8 ; j++ )
    //         {
    //             for ( int k = 0 ; k < 3 ; k++)
    //                 foo[GP][i][j] += E[i][k] * B[GP][k][j];
    //         }
    //     }
    // }

    
    // // bar = B^T * foo
    // for ( int GP = 0 ; GP < 4 ; GP++)
    // {
    //     for ( int i = 0 ; i < 8 ; i++ )
    //     {
    //         for( int j = 0 ; j < 8 ; j++ )
    //         {
    //             for ( int k = 0 ; k < 3 ; k++)
    //                 bar[GP][i][j] += B[GP][k][i] * foo[GP][k][j];
    //         }
    //     }
    // }


    // for ( int GP = 0 ; GP < 4 ; GP++)
    // {
    //     for ( int i = 0 ; i < 8 ; i++ )
    //     {
    //         for( int j = 0 ; j < 8 ; j++ )
    //             m_A_local[j + i*num_cols] += bar[GP][i][j];     // TODO: * jacobi here
    //     }
    // }


//     return A_local;
// }
