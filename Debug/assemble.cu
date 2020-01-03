#include <iostream>
#include "assemble.h"
#include <cmath>
#include "cudakernels.h"

using namespace std;


//// Node class definitions
Assembler::Node::Node (int id) : m_index(id){}
int Assembler::Node::index() { return m_index; }



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



Assembler::Assembler(size_t dim, double h, vector<size_t> N, double youngMod, double poisson, double rho, size_t p, size_t numLevels)
    : m_dim(dim), m_h(h), m_youngMod(youngMod), m_poisson(poisson), m_rho(rho), m_p(p), m_numLevels(numLevels)
{
    // cout << "assembler" << endl;

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
    // cout << "assembler : deallocate" << endl;
    // CUDA_CALL( cudaFree(d_m_A_local) );
}


void Assembler::setBC(vector<size_t> bc_index)
{
    m_bc_index = bc_index;
}

bool Assembler::init(
    double* &d_A_local, 
    vector<double*> &d_value, 
    vector<size_t*> &d_index, 
    vector<double*> &d_p_value, 
    vector<size_t*> &d_p_index, 
    double* &d_kai, 
    vector<size_t> &num_rows, 
    vector<size_t> &max_row_size, 
    vector<size_t> &p_max_row_size)
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
  
    num_rows.resize(m_numLevels);
    for ( int lev = 0 ; lev < m_numLevels ; lev++ )
        num_rows[lev] = m_numNodes[lev] * m_dim;


    // storing the design variable in each element
    // initial value is rho in all elements
    m_kai.resize(m_numElements[m_topLev], m_rho);

    // TODO: CHECK: assemblelocal is messed up a bit, recheck especially when it comes to the det(J)
    assembleLocal();

    // int a = 0;
    // for ( int j = 0 ; j < 8 ; j++ )
    // {
    //     for ( int i = 0 ; i < 8 ; i++ )
    //         {
    //             cout << m_A_local[a] << " ";
    //             a++;
    //         }

    //         cout << "\n";
    // }
    
    // TODO: CHECK: check the numbers here as well
    assembleGlobal(num_rows, max_row_size, p_max_row_size);

    
    //// CUDA

    // allocating memory in device

    // design variables
    CUDA_CALL( cudaMalloc((void**)&d_kai, sizeof(double) * m_numElements[m_topLev] ) );

    // local stiffness
    CUDA_CALL( cudaMalloc((void**)&d_A_local, sizeof(double) * m_A_local.size() ) );

    // prolongation matrices on each grid-level
    d_p_value.resize( m_numLevels - 1 );
    d_p_index.resize( m_numLevels - 1 );

    for ( int lev = 0 ; lev < m_numLevels - 1 ; lev++ )
    {
        CUDA_CALL( cudaMalloc((void**)&d_p_value[lev], sizeof(double) * p_max_row_size[lev] * num_rows[lev+1] ) );
        CUDA_CALL( cudaMalloc((void**)&d_p_index[lev], sizeof(size_t) * p_max_row_size[lev] * num_rows[lev+1] ) );
    }
    
    // global matrices on each grid-level
    d_value.resize( m_numLevels );
    d_index.resize( m_numLevels );

    for ( int lev = 0 ; lev < m_numLevels ; lev++ )
    {
        CUDA_CALL( cudaMalloc((void**)&d_value[lev], sizeof(double) * max_row_size[lev] * num_rows[lev] ) );
        CUDA_CALL( cudaMalloc((void**)&d_index[lev], sizeof(size_t) * max_row_size[lev] * num_rows[lev] ) );
    }

    cudaDeviceSynchronize();

    // copy memory to device
    
    CUDA_CALL( cudaMemcpy(d_A_local, &m_A_local[0], sizeof(double) * m_A_local.size(), cudaMemcpyHostToDevice) );
    CUDA_CALL( cudaMemcpy(d_kai, &m_kai[0], sizeof(double) * m_numElements[m_topLev], cudaMemcpyHostToDevice) );

    for ( int lev = 0 ; lev < m_numLevels - 1 ; lev++ )
    {
        CUDA_CALL( cudaMemcpy(d_p_value[lev], &m_p_value_g[lev][0], sizeof(double) * p_max_row_size[lev] * num_rows[lev+1], cudaMemcpyHostToDevice) );
        CUDA_CALL( cudaMemcpy(d_p_index[lev], &m_p_index_g[lev][0], sizeof(size_t) * p_max_row_size[lev] * num_rows[lev+1], cudaMemcpyHostToDevice) );
    }

    for ( int lev = 0 ; lev < m_numLevels ; lev++ )
    {
        CUDA_CALL( cudaMemcpy(d_value[lev], &m_value_g[lev][0], sizeof(double) * max_row_size[lev] * num_rows[lev], cudaMemcpyHostToDevice) );
        CUDA_CALL( cudaMemcpy(d_index[lev], &m_index_g[lev][0], sizeof(size_t) * max_row_size[lev] * num_rows[lev], cudaMemcpyHostToDevice) );
    }
    





    return true;

}


// TODO: check this is not right!
// assembles the local stiffness matrix
bool Assembler::assembleLocal()
{
    // cout << "assembleLocal" << endl;


    double E[3][3];

    E[0][0] = E[1][1] = m_youngMod/(1 - m_poisson * m_poisson );
    E[0][1] = E[1][0] = m_poisson * E[0][0];
    E[2][2] = (1 - m_poisson) / 2 * E[0][0];
    E[2][0] = E[2][1] = E[1][2] = E[0][2] = 0.0;

    // bilinear shape function matrix (using 4 Gauss Points)
    double B[4][3][8] = { { {-0.3943375,	0,	0.3943375,	0,	0.1056625,	0,	-0.1056625,	0}, {0,	-0.3943375,	0,	-0.1056625,	0,	0.1056625,	0,	0.3943375} , {-0.3943375,	-0.3943375,	-0.1056625,	0.3943375,	0.1056625,	0.1056625,	0.3943375,	-0.1056625} },
                          { {-0.3943375,	0,	0.3943375,	0,	0.1056625,	0,	-0.1056625,	0}, {0,	-0.1056625,	0,	-0.3943375,	0,	0.3943375,	0,	0.1056625}, {-0.1056625,	-0.3943375,	-0.3943375,	0.3943375,	0.3943375,	0.1056625,	0.1056625,	-0.1056625} },
                          { {-0.1056625,	0,	0.1056625,	0,	0.3943375,	0,	-0.3943375,	0}, {0,	-0.3943375,	0,	-0.1056625,	0,	0.1056625,	0,	0.3943375}, {-0.3943375,	-0.1056625,	-0.1056625,	0.1056625,	0.1056625,	0.3943375,	0.3943375,	-0.3943375} },
                          { {-0.1056625,	0,	0.1056625,	0,	0.3943375,	0,	-0.3943375,	0}, {0,	-0.1056625,	0,	-0.3943375,	0,	0.3943375,	0,	0.1056625}, {-0.1056625,	-0.1056625,	-0.3943375,	0.1056625,	0.3943375,	0.3943375,	0.1056625,	-0.3943375} }
                        };



    // applying jacobi
    // B = inv(J) * N'
    for ( int i = 0 ; i < 4 ; i++ )
    {
        for ( int j = 0 ; j < 3 ; j++ )
        {
            for ( int k = 0 ; k < 8 ; k++ )
                B[i][j][k] *= ( 2 / m_h );  // TODO: check if this formula is correct

        }
    }

    //// 4 matrices with size 3x8 to store each GP's stiffness matrix

    // foo as a temp vector
    vector<vector<vector<double>>> foo;
    foo.resize(4); // 4 Gauss Points
    

    for ( int GP = 0 ; GP < 4 ; GP++)
    {
        // number of rows in each level
        foo[GP].resize(3);
        
        // number of columns in each level
        for ( int j = 0 ; j < 3 ; j++ )
                foo[GP][j].resize(8, 0.0);
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

    // bar = B^T * foo * det(J)
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

    // storing the value of B^T * E * B for later use in TDO




    // TODO: insertion of these numbers has to be automated
    // multiplying det(J)
    // for ( int i = 0 ; i < 8 * 8 ; i++ )
    //     m_A_local[i] *= pow(0.25,m_dim);

    return true;
}


double Assembler::valueAt(size_t x, size_t y)
{
    return m_A_local[y + x*m_num_rows_l];
}

// to produce an ELLmatrix of the global stiffness in the device
// will return d_value, d_index, d_max_row_size
bool Assembler::assembleGlobal(vector<size_t> &num_rows, vector<size_t> &max_row_size, vector<size_t> &p_max_row_size)
{
    // TODO: if no BC is set, return false with error
    // cout << "assembleGlobal" << endl;
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

    // resizing the global stiffness matrices on each grid-level
    m_A_g.resize(m_numLevels);

    for ( int lev = 0 ; lev < m_numLevels ; lev++ )
    {
        // number of columns in each level
        m_A_g[lev].resize(num_rows[lev]);
        
        // number of rows in each level
        for ( int j = 0 ; j < num_rows[lev] ; j++ )
                m_A_g[lev][j].resize(num_rows[lev]);
    }

    // filling in the global stiffness matrix from the local stiffness matrices of the 4 Gauss-Points
    for ( int elmn_index = 0 ; elmn_index < 4 ; elmn_index++ )
    {
        for ( int x = 0 ; x < 4 ; x++ ) // TODO: dim  
        {
            for ( int y = 0 ; y < 4 ; y++ )        // TODO: dim   
            {      
                    m_A_g[m_topLev][ 2*m_element[elmn_index].nodeIndex(x)     ][ 2*m_element[elmn_index].nodeIndex(y)     ] += pow(m_rho, m_p) * valueAt( 2*x    , 2*y     );
                    m_A_g[m_topLev][ 2*m_element[elmn_index].nodeIndex(x)     ][ 2*m_element[elmn_index].nodeIndex(y) + 1 ] += pow(m_rho, m_p) * valueAt( 2*x    , 2*y + 1 );
                    m_A_g[m_topLev][ 2*m_element[elmn_index].nodeIndex(x) + 1 ][ 2*m_element[elmn_index].nodeIndex(y)     ] += pow(m_rho, m_p) * valueAt( 2*x + 1, 2*y     );
                    m_A_g[m_topLev][ 2*m_element[elmn_index].nodeIndex(x) + 1 ][ 2*m_element[elmn_index].nodeIndex(y) + 1 ] += pow(m_rho, m_p) * valueAt( 2*x + 1, 2*y + 1 );
            }
        }
    }

    // cleanup: replacing any values <1e-7 to 0.0
    for ( int x = 0 ; x < m_numNodes[m_topLev]*m_dim ; x++ ) // TODO: dim  
    {
        for ( int y = 0 ; y < m_numNodes[m_topLev]*m_dim ; y++ )        // TODO: dim   
        {      
            if ( m_A_g[m_topLev][x][y] < 1e-7 && m_A_g[m_topLev][x][y] > -1e-7)
                m_A_g[m_topLev][x][y] = 0.0;
        }
    }
    
    // applying BC on the matrix
    // DOFs which are affected by BC will have identity rows/cols { 0 0 .. 1 .. 0 0}
    for ( int i = 0 ; i < m_bc_index.size() ; ++i )
        applyMatrixBC(m_A_g[m_topLev], m_bc_index[i], num_rows[m_topLev]);

    // resizing the prolongation matrices according to the number of grid-levels
    m_P.resize( m_numLevels - 1 );
    
    for ( int lev = 0 ; lev < m_numLevels - 1 ; lev++)
    {
        // number of columns in each level
        m_P[lev].resize(num_rows[lev]);
        
        // number of rows in each level
        for ( int j = 0 ; j < num_rows[lev] ; j++ )
                m_P[lev][j].resize(num_rows[lev + 1]);
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

    // for ( int i = 0 ; i < m_num_rows[1] ; i++ )
    // {
    //     for ( int j = 0 ; j < m_num_rows[1] ; j++ )
    //         cout << m_A_g[1][i][j] << " ";

    //     cout << "\n";
    // }

    // // for ( int i = 0 ; i < 4 ; i++ )
    // // {
    // //     for ( int j = 0 ; j < 4 ; j++ )
    // //         cout << A_coarse[i][j] << " ";

    // //     cout << "\n";
    // // }

    // resizing the coarse stiffness matrices on each grid-level

    for ( int lev = 0 ; lev < m_numLevels - 1; lev++ )
        PTAP(m_A_g[lev], m_A_g[lev+1], m_P[lev], num_rows[lev+1], num_rows[lev] );

    max_row_size.resize(m_numLevels);
    p_max_row_size.resize(m_numLevels - 1);
    // calculate global max_num_rows, which will also be needed when allocating memory in device
    for ( int lev = 0 ; lev < m_numLevels ; lev++ )
        max_row_size[lev] = getMaxRowSize(m_A_g[lev], num_rows[lev], num_rows[lev]);
    
    for ( int lev = 0 ; lev < m_numLevels - 1 ; lev++ )
        p_max_row_size[lev] = getMaxRowSize(m_P[lev], num_rows[lev+1], num_rows[lev]);
    

    //// obtaining the ELLPACK value and index vectors from the global stiffness matrix
    
    // resizing the vectors
    m_p_value_g.resize( m_numLevels - 1 );
    m_p_index_g.resize( m_numLevels - 1 );
    m_value_g.resize( m_numLevels );
    m_index_g.resize( m_numLevels );

    // prolongation matrices
    for ( int lev = 0 ; lev < m_numLevels - 1 ; lev++ )
        transformToELL(m_P[lev], m_p_value_g[lev], m_p_index_g[lev], p_max_row_size[lev], num_rows[lev+1], num_rows[lev] );    

    // stiffness matrices
    for ( int lev = 0 ; lev < m_numLevels ; lev++ )
        transformToELL(m_A_g[lev], m_value_g[lev], m_index_g[lev], max_row_size[lev], num_rows[lev], num_rows[lev] );
        

    // int a = 0;
    // for ( int j = 0 ; j < m_num_rows[1] ; j++ )
    // {
    //     for ( int i = 0 ; i < m_max_row_size[1] ; i++ )
    //         {
    //             cout << m_value_g[1][a] << " ";
    //             a++;
    //         }

    //         cout << "\n";
    // }


    // NOTE: can somehow do init for solving now while allocating memory in device?
    // do async malloc then your init() should be AFTER the memcpy stuff, not before





    return true;

}

    // int a = 0;
    // for ( int j = 0 ; j < 8 ; j++ )
    // {
    //     for ( int i = 0 ; i < 8 ; i++ )
    //         {
    //             cout << m_A_local[a] << " ";
    //             a++;
    //         }

    //         cout << "\n";
    // }
