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

size_t Assembler::Element::getNodeIndex(int index)
{
    return m_node[index]->index();
}


Assembler::Assembler(size_t dim, double h, vector<size_t> N, double youngMod, double poisson, double rho, size_t p, size_t numLevels)
    : m_dim(dim), m_h(h), m_youngMod(youngMod), m_poisson(poisson), m_rho(rho), m_p(p), m_numLevels(numLevels)
{
    // cout << "assembler" << endl;

    // Check that dim and size of each dimension, N tally
    if ( dim != N.size() )
        throw(runtime_error("Error : N is not defined for each dimension"));
    

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


void Assembler::setBC(vector<vector<size_t>> bc_index)
{
    m_bc_index = bc_index;
}

size_t Assembler::getNumElements() 
{
    return m_numElements[m_topLev];
}

vector<size_t> Assembler::getNumNodesPerDim()
{
    return m_numNodesPerDim[m_topLev];
}

size_t Assembler::getNumNodes()
{
    return m_numNodes[m_topLev];
}

vector<size_t> Assembler::getGridSize()
{
    return m_N[m_topLev];
}


bool Assembler::init_GPU(
    double* &d_A_local, 
    vector<double*> &d_value, 
    vector<size_t*> &d_index, 
    vector<double*> &d_p_value, 
    vector<size_t*> &d_p_index, 
    vector<double*> &d_r_value, 
    vector<size_t*> &d_r_index, 
    double* &d_chi, 
    vector<size_t> &num_rows, 
    vector<size_t> &max_row_size, 
    vector<size_t> &p_max_row_size,
    vector<size_t> &r_max_row_size,
    vector<size_t*> &d_node_index)
{

   


    // TODO: CHECK:
    m_d_A_local = d_A_local;

    // setting size of the local stiffness matrix
    if ( m_dim == 2 )
    {
        m_A_local.resize(64, 0.0);
        m_num_rows_l = 8;
    }

    else if (m_dim == 3 )
    {
        m_A_local.resize(576, 0.0);
        m_num_rows_l = 24;
    }

    else
        throw(runtime_error("Error : Grid dimension must be 2D or 3D"));

    // getting the top level grid index
    m_topLev = m_numLevels - 1;

   
    // number of elements in each grid-level
    m_numElements.resize(m_numLevels, 1);
    
    for ( int lev = 0 ; lev < m_numLevels ; lev++ )
    {
        for ( int i = 0 ; i < m_dim ; i++)
        {
            m_numElements[lev] *= m_N[lev][i];
        }
    }
    
    // obtaining the maximum row size of each grid
    max_row_size.resize(m_numLevels);
    p_max_row_size.resize(m_numLevels - 1);
    r_max_row_size.resize(m_numLevels - 1);

    if ( m_topLev == 0 )
        throw(runtime_error("Error : Please use at least 2 levels of multigrid solver"));

    else
    {
        if ( m_dim == 2 )
        {

            // calculating the max row sizes of the global stiffness, prolongation and restriction matrices
            
            if (m_N[0][0] == 1 && m_N[0][1] == 1)
            {
                max_row_size[0] = 8;
                p_max_row_size[0] = 4;
                r_max_row_size[0] = 4;
            }
            else if (m_N[0][0] == 1 || m_N[0][1] == 1)
            {
                max_row_size[0] = 12;
                p_max_row_size[0] = 4;
                r_max_row_size[0] = 6;
            }
            else
            {
                max_row_size[0] = 18;
                p_max_row_size[0] = 4;
                r_max_row_size[0] = 9;
            }

            // loop through each level
            for (int lev = m_topLev ; lev != 0 ; --lev)
            {
                max_row_size[lev] = 18;
                p_max_row_size[lev] = 4;
                r_max_row_size[lev] = 9;
            }
        }

        else 
        {
            throw(runtime_error("TODO: calculate max_row_size for 3D not done yet!"));
        }
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

    // number of DOFs per level
    num_rows.resize(m_numLevels);
    for ( int lev = 0 ; lev < m_numLevels ; lev++ )
        num_rows[lev] = m_numNodes[lev] * m_dim;


    // storing the design variable in each element
    // initial value is rho in all elements
    m_chi.resize(m_numElements[m_topLev], m_rho);

    
    // calculate CUDA block dimension for Ellpack matrices
    m_ell_gridDim.resize(m_numLevels);
    m_ell_blockDim.resize(m_numLevels);

    for ( int i = 0 ; i < m_numLevels ; ++i )
        calculateDimensions(num_rows[i]*max_row_size[i], m_ell_gridDim[i], m_ell_blockDim[i]);

    m_num_rows = num_rows;
    m_max_row_size = max_row_size;
    m_r_max_row_size = r_max_row_size;
    m_p_max_row_size = p_max_row_size;
    
    cout << m_r_max_row_size[m_topLev] << endl;
    


    // DEBUG:

    // TODO: 3D doesn't work yet
    assembleLocal();

    
    //// prolongation & restriction matrices

    // resizing the prolongation matrices according to the number of grid-levels
    m_P.resize( m_numLevels - 1 );

    for ( int lev = 0 ; lev < m_numLevels - 1 ; lev++)
    {
        // number of columns in each level
        m_P[lev].resize(num_rows[lev+1]);
        
        // number of rows in each level
        for ( int j = 0 ; j < num_rows[lev+1] ; j++ )
                m_P[lev][j].resize(num_rows[lev]);
    }
    
    assembleProlMatrix(m_topLev);

    // TODO:
    // assembleProlMatrix_GPU(d_p_value, d_p_index, m_topLev);

//     // resizing the restriction matrices according to the number of grid-levels
//     m_R.resize( m_numLevels - 1 );

//     for ( int lev = 0 ; lev < m_numLevels - 1 ; lev++)
//     {
//         // number of columns in each level
//         m_R[lev].resize(num_rows[lev]);
        
//         // number of rows in each level
//         for ( int j = 0 ; j < num_rows[lev] ; j++ )
//                 m_R[lev][j].resize(num_rows[lev+1]);
//     }

//     assembleRestMatrix(m_topLev);
    assembleRestMatrix_GPU(d_r_value, d_r_index, m_topLev);


//     //// obtaining the ELLPACK value and index vectors from the global stiffness matrix

//     // resizing the vectors required for ELLPACK for each level
//     p_max_row_size.resize(m_numLevels - 1);
//     r_max_row_size.resize(m_numLevels - 1);

//     for ( int lev = 0 ; lev < m_numLevels - 1 ; lev++ )
//         p_max_row_size[lev] = getMaxRowSize(m_P[lev], num_rows[lev+1], num_rows[lev]);
    
//     for ( int lev = 0 ; lev < m_numLevels - 1 ; lev++ )
//         r_max_row_size[lev] = getMaxRowSize(m_R[lev], num_rows[lev], num_rows[lev+1]);
    
//     m_r_max_row_size = r_max_row_size;
//     m_p_max_row_size = p_max_row_size;
    

//     // resizing the vectors
//     m_p_value_g.resize( m_numLevels - 1 );
//     m_p_index_g.resize( m_numLevels - 1 );
//     m_r_value_g.resize( m_numLevels - 1 );
//     m_r_index_g.resize( m_numLevels - 1 );

//     // prolongation matrices
//     for ( int lev = 0 ; lev < m_numLevels - 1 ; lev++ )
//         transformToELL(m_P[lev], m_p_value_g[lev], m_p_index_g[lev], p_max_row_size[lev], num_rows[lev+1], num_rows[lev] );    

//     // restriction matrices
//     for ( int lev = 0 ; lev < m_numLevels - 1 ; lev++ )
//         transformToELL(m_R[lev], m_r_value_g[lev], m_r_index_g[lev], r_max_row_size[lev], num_rows[lev], num_rows[lev+1] );    

    

//     // prolongation matrices on each grid-level
//     d_p_value.resize( m_numLevels - 1 );
//     d_p_index.resize( m_numLevels - 1 );

//     for ( int lev = 0 ; lev < m_numLevels - 1 ; lev++ )
//     {
//         CUDA_CALL( cudaMalloc((void**)&d_p_value[lev], sizeof(double) * p_max_row_size[lev] * num_rows[lev+1] ) );
//         CUDA_CALL( cudaMalloc((void**)&d_p_index[lev], sizeof(size_t) * p_max_row_size[lev] * num_rows[lev+1] ) );
//     }

//     // restriction matrices on each grid-level
//     d_r_value.resize( m_numLevels - 1 );
//     d_r_index.resize( m_numLevels - 1 );



//     for ( int lev = 0 ; lev < m_numLevels - 1 ; lev++ )
//     {
//         CUDA_CALL( cudaMalloc((void**)&d_r_value[lev], sizeof(double) * r_max_row_size[lev] * num_rows[lev] ) );
//         CUDA_CALL( cudaMalloc((void**)&d_r_index[lev], sizeof(size_t) * r_max_row_size[lev] * num_rows[lev] ) );
//     }
    

//     for ( int lev = 0 ; lev < m_numLevels - 1 ; lev++ )
//     {
//         CUDA_CALL( cudaMemcpy(d_p_value[lev], &m_p_value_g[lev][0], sizeof(double) * p_max_row_size[lev] * num_rows[lev+1], cudaMemcpyHostToDevice) );
//         CUDA_CALL( cudaMemcpy(d_p_index[lev], &m_p_index_g[lev][0], sizeof(size_t) * p_max_row_size[lev] * num_rows[lev+1], cudaMemcpyHostToDevice) );
//     }

//     for ( int lev = 0 ; lev < m_numLevels - 1 ; lev++ )
//     {
//         CUDA_CALL( cudaMemcpy(d_r_value[lev], &m_r_value_g[lev][0], sizeof(double) * r_max_row_size[lev] * num_rows[lev], cudaMemcpyHostToDevice) );
//         CUDA_CALL( cudaMemcpy(d_r_index[lev], &m_r_index_g[lev][0], sizeof(size_t) * r_max_row_size[lev] * num_rows[lev], cudaMemcpyHostToDevice) );
//     }

//     // allocating temp matrix to be used in RAP
//     CUDA_CALL( cudaMalloc((void**)&m_d_temp_matrix, sizeof(double) * num_rows[m_topLev] * num_rows[m_topLev-1] ) );
//     CUDA_CALL( cudaMemset( m_d_temp_matrix, 0, sizeof(double) * num_rows[m_topLev] * num_rows[m_topLev-1] ) );
    


//     // adding nodes and elements to the top-level global grid
//     for ( int i = 0 ; i < m_numNodes[m_topLev] ; ++i )
//         m_node.push_back(Node(i));

//     for ( int i = 0 ; i < m_numElements[m_topLev] ; ++i )
//         m_element.push_back(Element(i));

//     size_t numNodesIn2D = (m_N[m_topLev][0]+1)*(m_N[m_topLev][1]+1);


//     // assign the nodes to each element
//     if ( m_dim == 2)
//     {
//         for ( int i = 0 ; i < m_numElements[m_topLev] ; i++ )
//         {
//             m_element[i].addNode(&m_node[ i + i/m_N[m_topLev][0] ]);   // lower left node
//             m_element[i].addNode(&m_node[ i + i/m_N[m_topLev][0] + 1]);   // lower right node
//             m_element[i].addNode(&m_node[ i + i/m_N[m_topLev][0] + m_N[m_topLev][0] + 1]);   // upper left node
//             m_element[i].addNode(&m_node[ i + i/m_N[m_topLev][0] + m_N[m_topLev][0] + 2]);   // upper right node
//         }
//     }

//     // m_dim == 3
//     else    
//     {
//         throw(runtime_error("3D not done yet!"));
//     }

//     m_node_index.resize(m_numElements[m_topLev]);
//     d_node_index.resize(m_numElements[m_topLev]);
//     for ( int elem = 0 ; elem < m_numElements[m_topLev] ; elem++ )
//     {
//         for ( int index = 0 ; index < pow(2, m_dim) ; index++ )
//             m_node_index[elem].push_back( m_element[elem].getNodeIndex(index) );
//     }


//     // design variable
//     CUDA_CALL( cudaMalloc((void**)&d_chi, sizeof(double) * m_numElements[m_topLev] ) );
//     CUDA_CALL( cudaMemcpy(d_chi, &m_chi[0], sizeof(double) * m_numElements[m_topLev], cudaMemcpyHostToDevice) );

//     // local stiffness matrix to device
//     CUDA_CALL( cudaMalloc((void**)&d_A_local, sizeof(double) * m_num_rows_l*m_num_rows_l ) );
//     CUDA_CALL( cudaMemcpy( d_A_local, &m_A_local[0], sizeof(double) * m_num_rows_l*m_num_rows_l, cudaMemcpyHostToDevice) );


//     // global matrices on each grid-level
//     d_value.resize( m_numLevels );
//     d_index.resize( m_numLevels );
    
//     // calculate the number of nodes in a local element
//     size_t numNodes_local = pow(2,m_dim);

//     for ( int i = 0 ; i < m_numElements[m_topLev] ; i++ )
//     {
//         CUDA_CALL( cudaMalloc( (void**)&d_node_index[i], sizeof(size_t) * numNodes_local) );
//         CUDA_CALL( cudaMemcpy( d_node_index[i], &m_node_index[i][0], sizeof(size_t) * numNodes_local, cudaMemcpyHostToDevice) );
//     }

//     m_d_node_index = d_node_index;

//     for ( int lev = 0 ; lev < m_numLevels ; lev++ )
//     {
//         CUDA_CALL( cudaMalloc((void**)&d_value[lev], sizeof(double) * max_row_size[lev] * num_rows[lev] ) );
//         CUDA_CALL( cudaMemset( d_value[lev], 0, sizeof(double) * num_rows[lev]*max_row_size[lev] ) );
//         CUDA_CALL( cudaMalloc((void**)&d_index[lev], sizeof(size_t) * max_row_size[lev] * num_rows[lev] ) );
//         CUDA_CALL( cudaMemset( d_index[lev], 0, sizeof(size_t) * num_rows[lev]*max_row_size[lev] ) );
//     }
    


//     // TODO: parallelizable
//     // fill in global stiffness matrix's ELLPACK index vector for all levels
//     dim3 index_gridDim;
//     dim3 index_blockDim;
//     for (int lev = m_topLev ; lev >= 0 ; --lev )
//     {
//         calculateDimensions( num_rows[lev], index_gridDim, index_blockDim);
//         fillIndexVector_GPU<<<index_gridDim,index_blockDim>>>(d_index[lev], m_N[lev][0], m_N[lev][1], max_row_size[lev], num_rows[lev]);
//     }

    
//     // assembleGlobal_GPU<<<1,num_rows[m_topLev-1]>>>(d_index[m_topLev-1], m_N[m_topLev-1][0], m_N[m_topLev-1][1], max_row_size[m_topLev-1], num_rows[m_topLev-1]);
//     cudaDeviceSynchronize();
    


//     // CUDA block size for assembling the global stiffness matrix
//     dim3 l_blockDim(m_num_rows_l,m_num_rows_l,1);

//     // cout << m_N[m_topLev][0] << endl;

//     for ( int i = 0 ; i < m_numElements[m_topLev] ; ++i )
//         assembleGrid2D_GPU<<<1,l_blockDim>>>( m_N[m_topLev][0], m_dim, &d_chi[i], d_A_local, &d_value[m_topLev][0], &d_index[m_topLev][0], max_row_size[m_topLev], m_num_rows_l, d_node_index[i], m_p);

// // printELLrow(1, d_value[1], d_index[1], max_row_size[1], num_rows[1], num_rows[1]);

//     // calculating the needed cuda 2D grid size for the global assembly
//     dim3 g_gridDim;
//     dim3 g_blockDim;

//     // for ( int i = 0 ; i < m_bc_index[m_topLev].size() ; i++ )
//     // cout << m_bc_index[m_topLev][i] << endl;
    
//     // TODO: CHECK: this is a bit shaky
//     // TODO: think it's a bit overkill to use a lot of cuda threads here
//     //// apply boundary condition to global and P/R matrices
//     // global stiffness matrix
//     calculateDimensions2D( num_rows[m_topLev], num_rows[m_topLev], g_gridDim, g_blockDim);
//     for ( int i = 0 ; i < m_bc_index[m_topLev].size() ; i++ )
//         applyMatrixBC_GPU_test<<<g_gridDim,g_blockDim>>>(&d_value[m_topLev][0], &d_index[m_topLev][0], max_row_size[m_topLev], m_bc_index[m_topLev][i], num_rows[m_topLev], num_rows[m_topLev] );


//     // // prolongation matrix
//     // calculateDimensions2D( num_rows[m_topLev-1], num_rows[m_topLev], g_gridDim, g_blockDim);
//     // for ( int i = 0 ; i < m_bc_index[m_topLev].size() ; i++ )
//     //     applyMatrixBC_GPU_test<<<g_gridDim,g_blockDim>>>(&d_p_value[m_topLev-1][0], &d_p_index[m_topLev-1][0], p_max_row_size[m_topLev-1], m_bc_index[m_topLev][i], num_rows[m_topLev], num_rows[m_topLev-1] );


//     // // TODO: use optimized matrix multiplication
//     dim3 temp_gridDim;
//     dim3 temp_blockDim;
       
//     // // d_temp_matrix[lev-1][lev] to store R*A
//     // double* d_temp_matrix;
//     // CUDA_CALL( cudaMalloc((void**)&d_temp_matrix, sizeof(double) * num_rows[m_topLev] * num_rows[m_topLev-1] ) );
//     // CUDA_CALL( cudaMemset( d_temp_matrix, 0, sizeof(double) * num_rows[m_topLev] * num_rows[m_topLev-1] ) );
    

//     // A_coarse = R * A_fine * P
//     for ( int lev = m_topLev ; lev != 0 ; lev--)
//     {
//         calculateDimensions(num_rows[lev] * num_rows[lev-1], temp_gridDim, temp_blockDim);
//         setToZero<<<temp_gridDim, temp_blockDim>>>( m_d_temp_matrix, num_rows[lev] * num_rows[lev-1]);
//         RAP( d_value, d_index, max_row_size, d_r_value, d_r_index, r_max_row_size, d_p_value, d_p_index, p_max_row_size, m_d_temp_matrix, num_rows, lev);
//     }




    // cout << "aps" << endl;
    // printELLrow(0, d_value[0], d_index[0], max_row_size[0], num_rows[0], num_rows[0]);
    // printELLrow(1, d_value[1], d_index[1], max_row_size[1], num_rows[1], num_rows[1]);
    // printELLrow(2, d_value[2], d_index[2], max_row_size[2], num_rows[2], num_rows[2]);
    // printELLrow(0, d_r_value[0], d_r_index[0], r_max_row_size[0], num_rows[0], num_rows[1]);
    // printELLrow(1, d_r_value[1], d_r_index[1], r_max_row_size[1], num_rows[1], num_rows[2]);
    // printELLrow(0, d_p_value[0], d_p_index[0], p_max_row_size[0], num_rows[1], num_rows[0]);
    // printELLrow(1, d_p_value[1], d_p_index[1], p_max_row_size[1], num_rows[2], num_rows[1]);

    // printVector_GPU<<<1,10>>>( dt_index, 10 );
    // printLinearVector( d_index[0], num_rows[0], max_row_size[0]);
    // printLinearVector( d_index[1], num_rows[1], max_row_size[1]);
    // printLinearVector( d_index[2], num_rows[2], max_row_size[2]);
    // printLinearVector( d_A_local, 8, 8);
    // printLinearVector( d_temp_matrix, 16, 42);
    cudaDeviceSynchronize();

    return true;

}





// TODO:
bool Assembler::assembleLocal()
{
    // DEBUG:

    double foo; // jacobi = foo * identity matrix // TODO: delete cmnt
    double det_jacobi;
    double inv_jacobi;
    vector<vector<double>> N;

    if ( m_dim == 2 )
    {
        vector<vector<double>> E (3, vector <double> (3, 0.0));
        vector<vector<double>> A_ (3, vector <double> (8, 0.0));

        E[0][0] = E[1][1] = m_youngMod/(1 - m_poisson * m_poisson );
        E[0][1] = E[1][0] = m_poisson * E[0][0];
        E[2][2] = (1 - m_poisson) / 2 * E[0][0];
        E[2][0] = E[2][1] = E[1][2] = E[0][2] = 0.0;

        // 4 gauss points
        vector<vector<double>> GP = {   {-0.57735,	-0.57735} ,
                                        { 0.57735,	-0.57735} ,
                                        {-0.57735,	 0.57735} ,
                                        { 0.57735,	 0.57735}
                                    };

        foo = m_h / 2 ;
        det_jacobi = pow(m_h/2, m_dim);
        inv_jacobi = 1 / det_jacobi;
        
        
        // loop through each set of gauss points
        for ( int i = 0 ; i < 4 ; ++i )
        {
            // resetting of vectors for each loop calculation
            N.resize(2, vector<double>(4));
            A_.clear();
            A_.resize(3, vector<double>(8));

            // bilinear element 
            N = {   { -(1-GP[i][1]),  (1-GP[i][1]), (1+GP[i][1]), -(1+GP[i][1]) } , 
                    { -(1-GP[i][0]), -(1+GP[i][0]), (1+GP[i][0]),  (1-GP[i][0]) } };
            
            vector<vector<double>> B(3, vector <double> (8, 0));
                    
            // node 0
            B[0][0] = -0.5*(1-GP[i][1])/m_h;
            B[2][1] = B[0][0];
            B[1][1] = -0.5*(1-GP[i][0])/m_h;
            B[2][0] = B[1][1];
        
            // node 1
            B[0][2] = 0.5*(1-GP[i][1])/m_h;
            B[2][3] = B[0][2];
            B[1][3] = -0.5*(1+GP[i][0])/m_h;
            B[2][2] = B[1][3];

            // node 2
            B[0][4] = -0.5*(1+GP[i][1])/m_h;
            B[2][5] = B[0][4];
            B[1][5] = 0.5*(1-GP[i][0])/m_h;
            B[2][4] = B[1][5];

            // node 3
            B[0][6] = 0.5*(1+GP[i][1])/m_h;
            B[2][7] = B[0][6];
            B[1][7] = 0.5*(1+GP[i][0])/m_h;
            B[2][6] = B[1][7];


            //// A_local = B^T * E * B * det(J)
            
            // A_ = E * B
            for ( int i = 0 ; i < 3 ; i++ )
            {
                for( int j = 0 ; j < 8 ; j++ )
                {
                    for ( int k = 0 ; k < 3 ; k++)
                        A_[i][j] += E[i][k] * B[k][j];
                }
            }

            // for ( int i = 0 ; i < 1 ; i++ )
            // {
            //     for( int j = 0 ; j < 1 ; j++ )
            //     {
            //         for ( int k = 0 ; k < 3 ; k++)
            //             printf("%f\n", E[i][k]);
            //             // A_[i][j] += E[i][k] * B[k][j];
            //     }
            // }
            
            // cout << A_[0][0] << endl;

            // for ( int i = 0 ; i < 3 ; i++ )
            // {
            //     for ( int j = 0 ; j < 8 ; j++ )
            //         cout << A_[i][j] << " ";

            //         cout << "\n";
            // }
            //         cout << "\n";
            
            // A_local = B^T * A_ * det(J)
            for ( int i = 0 ; i < 8 ; i++ )
            {
                for( int j = 0 ; j < 8 ; j++ )
                {
                    for ( int k = 0 ; k < 3 ; k++){
                        
                        m_A_local[j + i*m_num_rows_l] += B[k][i] * A_[k][j] * det_jacobi;  
                    }
                }
            }

        }

    }

    if ( m_dim == 3 )
    {
    // TODO: create function for this
    

    // isotropic linear elastic tensor
    double lambda = (m_youngMod * m_poisson) / ((1+m_poisson)*(1-2*m_poisson));
    double mu = m_youngMod / ( 2 * (1+m_poisson) );

    vector<vector<double>> E (6, vector <double> (6, 0.0));

    E[0][0] = E[1][1] = E[2][2] = lambda + 2*mu;
    E[1][0] = E[2][0] = E[0][1] = E[0][2] = E[1][2] = E[2][1] = lambda;
    E[3][3] = E[4][4] = E[5][5] = mu;

    m_A_local = {   
                37064000000,	10736000000,	10736000000,	-14314000000,	-639420000,	-639420000,	7157100000,	639420000,	5367800000,	-12845000000,	-10736000000,	-319710000,	7157100000,	5367800000,	639420000,	-12845000000,	-319710000,	-10736000000,	-2109000000,	319710000,	319710000,	-9266000000,	-5367800000,	-5367800000,
                10736000000,	37064000000,	10736000000,	639420000,	7157100000,	5367800000,	-639420000,	-14314000000,	-639420000,	-10736000000,	-12845000000,	-319710000,	5367800000,	7157100000,	639420000,	319710000,	-2109000000,	319710000,	-319710000,	-12845000000,	-10736000000,	-5367800000,	-9266000000,	-5367800000,
                10736000000,	10736000000,	37064000000,	639420000,	5367800000,	7157100000,	5367800000,	639420000,	7157100000,	319710000,	319710000,	-2109000000,	-639420000,	-639420000,	-14314000000,	-10736000000,	-319710000,	-12845000000,	-319710000,	-10736000000,	-12845000000,	-5367800000,	-5367800000,	-9266000000,
                -14314000000,	639420000,	639420000,	37064000000,	-10736000000,	-10736000000,	-12845000000,	10736000000,	319710000,	7157100000,	-639420000,	-5367800000,	-12845000000,	319710000,	10736000000,	7157100000,	-5367800000,	-639420000,	-9266000000,	5367800000,	5367800000,	-2109000000,	-319710000,	-319710000,
                -639420000,	7157100000,	5367800000,	-10736000000,	37064000000,	10736000000,	10736000000,	-12845000000,	-319710000,	639420000,	-14314000000,	-639420000,	-319710000,	-2109000000,	319710000,	-5367800000,	7157100000,	639420000,	5367800000,	-9266000000,	-5367800000,	319710000,	-12845000000,	-10736000000,
                -639420000,	5367800000,	7157100000,	-10736000000,	10736000000,	37064000000,	-319710000,	319710000,	-2109000000,	-5367800000,	639420000,	7157100000,	10736000000,	-319710000,	-12845000000,	639420000,	-639420000,	-14314000000,	5367800000,	-5367800000,	-9266000000,	319710000,	-10736000000,	-12845000000,
                7157100000,	-639420000,	5367800000,	-12845000000,	10736000000,	-319710000,	37064000000,	-10736000000,	10736000000,	-14314000000,	639420000,	-639420000,	-2109000000,	-319710000,	319710000,	-9266000000,	5367800000,	-5367800000,	7157100000,	-5367800000,	639420000,	-12845000000,	319710000,	-10736000000,
                639420000,	-14314000000,	639420000,	10736000000,	-12845000000,	319710000,	-10736000000,	37064000000,	-10736000000,	-639420000,	7157100000,	-5367800000,	319710000,	-12845000000,	10736000000,	5367800000,	-9266000000,	5367800000,	-5367800000,	7157100000,	-639420000,	-319710000,	-2109000000,	-319710000,
                5367800000,	-639420000,	7157100000,	319710000,	-319710000,	-2109000000,	10736000000,	-10736000000,	37064000000,	639420000,	-5367800000,	7157100000,	-319710000,	10736000000,	-12845000000,	-5367800000,	5367800000,	-9266000000,	-639420000,	639420000,	-14314000000,	-10736000000,	319710000,	-12845000000,
                -12845000000,	-10736000000,	319710000,	7157100000,	639420000,	-5367800000,	-14314000000,	-639420000,	639420000,	37064000000,	10736000000,	-10736000000,	-9266000000,	-5367800000,	5367800000,	-2109000000,	319710000,	-319710000,	-12845000000,	-319710000,	10736000000,	7157100000,	5367800000,	-639420000,
                -10736000000,	-12845000000,	319710000,	-639420000,	-14314000000,	639420000,	639420000,	7157100000,	-5367800000,	10736000000,	37064000000,	-10736000000,	-5367800000,	-9266000000,	5367800000,	-319710000,	-12845000000,	10736000000,	319710000,	-2109000000,	-319710000,	5367800000,	7157100000,	-639420000,
                -319710000,	-319710000,	-2109000000,	-5367800000,	-639420000,	7157100000,	-639420000,	-5367800000,	7157100000,	-10736000000,	-10736000000,	37064000000,	5367800000,	5367800000,	-9266000000,	319710000,	10736000000,	-12845000000,	10736000000,	319710000,	-12845000000,	639420000,	639420000,	-14314000000,
                7157100000,	5367800000,	-639420000,	-12845000000,	-319710000,	10736000000,	-2109000000,	319710000,	-319710000,	-9266000000,	-5367800000,	5367800000,	37064000000,	10736000000,	-10736000000,	-14314000000,	-639420000,	639420000,	7157100000,	639420000,	-5367800000,	-12845000000,	-10736000000,	319710000,
                5367800000,	7157100000,	-639420000,	319710000,	-2109000000,	-319710000,	-319710000,	-12845000000,	10736000000,	-5367800000,	-9266000000,	5367800000,	10736000000,	37064000000,	-10736000000,	639420000,	7157100000,	-5367800000,	-639420000,	-14314000000,	639420000,	-10736000000,	-12845000000,	319710000,
                639420000,	639420000,	-14314000000,	10736000000,	319710000,	-12845000000,	319710000,	10736000000,	-12845000000,	5367800000,	5367800000,	-9266000000,	-10736000000,	-10736000000,	37064000000,	-639420000,	-5367800000,	7157100000,	-5367800000,	-639420000,	7157100000,	-319710000,	-319710000,	-2109000000,
                -12845000000,	319710000,	-10736000000,	7157100000,	-5367800000,	639420000,	-9266000000,	5367800000,	-5367800000,	-2109000000,	-319710000,	319710000,	-14314000000,	639420000,	-639420000,	37064000000,	-10736000000,	10736000000,	-12845000000,	10736000000,	-319710000,	7157100000,	-639420000,	5367800000,
                -319710000,	-2109000000,	-319710000,	-5367800000,	7157100000,	-639420000,	5367800000,	-9266000000,	5367800000,	319710000,	-12845000000,	10736000000,	-639420000,	7157100000,	-5367800000,	-10736000000,	37064000000,	-10736000000,	10736000000,	-12845000000,	319710000,	639420000,	-14314000000,	639420000,
                -10736000000,	319710000,	-12845000000,	-639420000,	639420000,	-14314000000,	-5367800000,	5367800000,	-9266000000,	-319710000,	10736000000,	-12845000000,	639420000,	-5367800000,	7157100000,	10736000000,	-10736000000,	37064000000,	319710000,	-319710000,	-2109000000,	5367800000,	-639420000,	7157100000,
                -2109000000,	-319710000,	-319710000,	-9266000000,	5367800000,	5367800000,	7157100000,	-5367800000,	-639420000,	-12845000000,	319710000,	10736000000,	7157100000,	-639420000,	-5367800000,	-12845000000,	10736000000,	319710000,	37064000000,	-10736000000,	-10736000000,	-14314000000,	639420000,	639420000,
                319710000,	-12845000000,	-10736000000,	5367800000,	-9266000000,	-5367800000,	-5367800000,	7157100000,	639420000,	-319710000,	-2109000000,	319710000,	639420000,	-14314000000,	-639420000,	10736000000,	-12845000000,	-319710000,	-10736000000,	37064000000,	10736000000,	-639420000,	7157100000,	5367800000,
                319710000,	-10736000000,	-12845000000,	5367800000,	-5367800000,	-9266000000,	639420000,	-639420000,	-14314000000,	10736000000,	-319710000,	-12845000000,	-5367800000,	639420000,	7157100000,	-319710000,	319710000,	-2109000000,	-10736000000,	10736000000,	37064000000,	-639420000,	5367800000,	7157100000,
                -9266000000,	-5367800000,	-5367800000,	-2109000000,	319710000,	319710000,	-12845000000,	-319710000,	-10736000000,	7157100000,	5367800000,	639420000,	-12845000000,	-10736000000,	-319710000,	7157100000,	639420000,	5367800000,	-14314000000,	-639420000,	-639420000,	37064000000,	10736000000,	10736000000,
                -5367800000,	-9266000000,	-5367800000,	-319710000,	-12845000000,	-10736000000,	319710000,	-2109000000,	319710000,	5367800000,	7157100000,	639420000,	-10736000000,	-12845000000,	-319710000,	-639420000,	-14314000000,	-639420000,	639420000,	7157100000,	5367800000,	10736000000,	37064000000,	10736000000,
                -5367800000,	-5367800000,	-9266000000,	-319710000,	-10736000000,	-12845000000,	-10736000000,	-319710000,	-12845000000,	-639420000,	-639420000,	-14314000000,	319710000,	319710000,	-2109000000,	5367800000,	639420000,	7157100000,	639420000,	5367800000,	7157100000,	10736000000,	10736000000,	37064000000
                };
    }

    return true;
}

// TODO: 3D is not done
// assembles the local stiffness matrix
bool Assembler::assembleLocal_()
{
    // cout << "assembleLocal" << endl;

    if ( m_dim == 2 )
    {
    vector<vector<double>> E (3, vector <double> (3, 0.0));

    E[0][0] = E[1][1] = m_youngMod/(1 - m_poisson * m_poisson );
    E[0][1] = E[1][0] = m_poisson * E[0][0];
    E[2][2] = (1 - m_poisson) / 2 * E[0][0];
    E[2][0] = E[2][1] = E[1][2] = E[0][2] = 0.0;

    // TODO: create function for this
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

    }

    else if (m_dim == 3)
    {
    // TODO: create function for this
    

    // isotropic linear elastic tensor
    double lambda = (m_youngMod * m_poisson) / ((1+m_poisson)*(1-2*m_poisson));
    double mu = m_youngMod / ( 2 * (1+m_poisson) );

    vector<vector<double>> E (6, vector <double> (6, 0.0));

    E[0][0] = E[1][1] = E[2][2] = lambda + 2*mu;
    E[1][0] = E[2][0] = E[0][1] = E[0][2] = E[1][2] = E[2][1] = lambda;
    E[3][3] = E[4][4] = E[5][5] = mu;

    m_A_local = {   
                    31444764,	10282982,	11217839,	-10982083,	1682940,	3085001,	6405116,	-2384040.85,	4206600,	-8576248,	-8880806,	841338,	6883005,	4954499,	-4487180,	-15209999,	-1497989,	-8804801,	-1544100,	-1822853,	-841328,	-6754910,	-4580610,	-4206670,
                    10282982,	30128660,	10282842,	-1682524.7,	3628424,	3271792,	981339,	-11235484.3,	2149959,	-8880810,	-8983000,	373890,	4393671,	9367130,	-2430550.6,	-2769843,	-4465050,	-2252358,	420670,	-9611226,	-8319625,	-4300160,	-6815500,	-4300120,
                    11217839,	10282842,	27393766,	-280304,	3271855,	5221124,	4206600,	-1215423.77,	5781840,	-841338,	-1308766,	-474390,	-1121829,	1495603,	-10099304.1,	-8996331,	-1553591,	-10056840,	841342,	-8039401,	-9334272,	-4206660,	-4580590,	-6443260,
                    -10982083,	-1682524.7,	-280304,	22967980,	-7478423.9,	-5608784,	-8575927,	7945788,	-841327,	6779238,	514032,	-4206600,	-4535495,	-934800,	7010910,	3448260,	-3451309,	-68906,	-6754975,	3832760,	4206650,	-1170260,	327174,	841313,
                    1682940,	3628424,	3271855,	-7478423.9,	25703212,	8413287.3,	7945689,	-8982883,	373886,	-2851182,	-9926211,	2149896,	467402,	-1151200,	-1028318.4,	-3604065,	5019018,	-3891012,	4113116,	-6815600,	-4300110,	-794595,	-8302469,	-8319699,
                    3085001,	3271792,	5221124,	-5608784,	8413287.3,	24526357,	841327,	-1308774,	-474290,	-4206600,	-1215343,	6155870,	7010920,	93462.0000000001,	-8118950,	-3242773,	-1956178,	-7460300,	4206670,	-4580550,	-6443320,	-841371,	-8039345,	-8960206.3,
                    6405116,	981339,	4206600,	-8575927,	7945689,	841327,	22220370,	-7478654,	5608784,	-9860301,	-747641,	280304,	-1544170,	-140254,	-841313,	-5969993,	3171697,	-3935820,	1523300,	-4393720,	1121906,	-3413761,	38,	-7010910,
                    -2384040.85,	-11235484.3,	-1215423.77,	7945788,	-8982883,	-1308774,	-7478654,	23087030,	-8413547,	2617749,	7554920,	-5141489,	-1262025,	-9611245.78,	8506767,	3607750,	-5089830,	3589989,	-4954655,	3228646,	2804618,	1402194.3,	2775008,	654401,
                    4206600,	2149959,	5781840,	-841327,	373886,	-474290,	5608784,	-8413547,	23778752,	-3085001,	-5141516,	6343020,	841371,	8787219,	-9334185,	-3987160,	3261271,	-5571720,	4487273,	95,	-12654492,	-7010920,	-1589150,	-6997097,
                    -8576248,	-8880810,	-841338,	6779238,	-2851182,	-4206600,	-9860301,	2617749,	-3085001,	29949602,	8413391,	-11217839,	-6754890,	-4580640,	4206670,	409640,	-1825186,	1123468,	-15753638,	1682726,	9815521,	5387330,	4954463,	4487180,
                    -8880806,	-8983000,	-1308766,	514032,	-9926211,	-1215343,	-747641,	7554920,	-5141516,	8413391,	24893809,	-6543572,	-4300130,	-6815510,	4113180,	417168,	-4098527,	6885921,	-280476,	-2553390,	654408,	4393620,	4132330,	934794,
                    841338,	373890,	-474390,	-4206600,	2149896,	6155870,	280304,	-5141489,	6343020,	-11217839,	-6543572,	25897614,	4206660,	3832769,	-6443350,	-507872,	6875702.6,	-6504515.97,	9815531,	-1589188,	-10923377.7,	1121829,	-1869609,	-11594948,
                    6883005,	4393671,	-1121829,	-4535495,	467402,	7010920,	-1544170,	-1262025,	841371,	-6754890,	-4300130,	4206660,	20226184,	5608803,	-5608759,	-10743738,	-2837187,	1428093,	7527120,	420842,	-4206661,	-8949910,	-7478288,	-841321,
                    4954499,	9367130,	1495603,	-934800,	-1151200,	93462.0000000001,	-140254,	-9611245.78,	8787219,	-4580640,	-6815510,	3832769,	5608803,	28726330,	-9347993,	-4611618,	-1009390,	-1622523,	3786156,	-7308473.6,	-2617498.5,	-7478328,	-10291465,	93481,
                    -4487180,	-2430550.6,	-10099304.1,	7010910,	-1028318.4,	-8118950,	-841313,	8506767,	-9334185,	4206670,	4113180,	-6443350,	-5608759,	-9347993,	24588990,	4078570,	-1029653,	3072928,	-4206651,	747829.1,	6903920,	841321,	1776112,	-848247,
                    -15209999,	-2769843,	-8996331,	3448260,	-3604065,	-3242773,	-5969993,	3607750,	-3987160,	409640,	417168,	-507872,	-10743738,	-4611618,	4078570,	24956767,	993209,	5579789,	-7120763.4,	8106331,	1107577,	8194410,	3966832,	3387451,
                    -1497989,	-4465050,	-1553591,	-3451309,	5019018,	-1956178,	3171697,	-5089830,	3261271,	-1825186,	-4098527,	6875702.6,	-2837187,	-1009390,	-1029653,	993209,	18209501,	-2781794,	7790700,	-5355648.9,	-41983,	1594846,	-1158936,	-2297584,
                    -8804801,	-2252358,	-10056840,	-68906,	-3891012,	-7460300,	-3935820,	3589989,	-5571720,	1123468,	6885921,	-6504515.97,	1428093,	-1622523,	3072928,	5579789,	-2781794,	17763904,	-383554,	1238953,	341440,	3195941,	-18777,	7248070,
                    -1544100,	420670,	841342,	-6754975,	4113116,	4206670,	1523300,	-4954655,	4487273,	-15753638,	-280476,	9815531,	7527120,	3786156,	-4206651,	-7120763.4,	7790700,	-383554,	33439321,	-8413486,	-11217889,	-9486684,	-4019806,	-3085000,
                    -1822853,	-9611226,	-8039401,	3832760,	-6815600,	-4580550,	-4393720,	3228646,	95,	1682726,	-2553390,	-1589188,	420842,	-7308473.6,	747829.1,	8106331,	-5355648.9,	1238953,	-8413486,	24488689,	5608895,	-654492,	8863400,	6076272,
                    -841328,	-8319625,	-9334272,	4206650,	-4300110,	-6443320,	1121906,	2804618,	-12654492,	9815521,	654408,	-10923377.7,	-4206661,	-2617498.5,	6903920,	1107577,	-41983,	341440,	-11217889,	5608895,	26583847,	280297,	6076275,	6716730,
                    -6754910,	-4300160,	-4206660,	-1170260,	-794595,	-841371,	-3413761,	1402194.3,	-7010920,	5387330,	4393620,	1121829,	-8949910,	-7478328,	841321,	8194410,	1594846,	3195941,	-9486684,	-654492,	280297,	17982766,	7478364,	5608759,
                    -4580610,	-6815500,	-4580590,	327174,	-8302469,	-8039345,	38,	2775008,	-1589150,	4954463,	4132330,	-1869609,	-7478288,	-10291465,	1776112,	3966832,	-1158936,	-18777,	-4019806,	8863400,	6076275,	7478364,	20873862,	7478474,
                    -4206670,	-4300120,	-6443260,	841313,	-8319699,	-8960206.3,	-7010910,	654401,	-6997097,	4487180,	934794,	-11594948,	-841321,	93481,	-848247,	3387451,	-2297584,	7248070,	-3085000,	6076272,	6716730,	5608759,	7478474,	22345282};

    }
    
    else
    {
        cout << "Error : dim must be 2 or 3" << endl;
        return false;
    }



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


bool Assembler::assembleProlMatrix_GPU(vector<double*> &d_p_value, vector<size_t*> &d_p_index, size_t lev)
{
    // resizing the prolongation matrices according to the number of grid-levels
    d_p_value.resize( m_numLevels - 1 );
    d_p_index.resize( m_numLevels - 1 );

    // for ( int lev = 0 ; lev < m_numLevels - 1 ; lev++)
    // {
    //     // number of columns in each level
    //     m_P[lev].resize(num_rows[lev+1]);
        
    //     // number of rows in each level
    //     for ( int j = 0 ; j < num_rows[lev+1] ; j++ )
    //             m_P[lev][j].resize(num_rows[lev]);
    // }

    //     // resizing the vectors
    // m_p_value_g.resize( m_numLevels - 1 );
    // m_p_index_g.resize( m_numLevels - 1 );
    // m_r_value_g.resize( m_numLevels - 1 );
    // m_r_index_g.resize( m_numLevels - 1 );
    return true;
}

bool Assembler::assembleProlMatrix(size_t lev)
{

    for ( int k = lev ; k != 0 ; k-- )
    {
        for ( int i = 0 ; i < m_numNodes[k-1] ; i++ )
        {
            for ( int j = 0 ; j < m_dim ; j++ )
            {

            // same node
                m_P[k-1][ m_dim*getFineNode(i, m_N[k-1], m_dim) + j ][m_dim*i+j] = 1;

            // east node
            if ( i % (m_N[k-1][0] + 1) != m_N[k-1][0] )
                m_P[k-1][ m_dim*getFineNode(i, m_N[k-1], m_dim) + j + m_dim ][m_dim*i+j] += 0.5;
            
            // north node
            if ( i % ((m_N[k-1][0]+1)*(m_N[k-1][1]+1)) < (m_N[k-1][0] + 1)*(m_N[k-1][1]) )
                m_P[k-1][ m_dim*getFineNode(i, m_N[k-1], m_dim) + j + (m_dim*( 2*(m_N[k][0] + 1) ))/2 ][m_dim*i+j] += 0.5;

            // west node
            if ( i % (m_N[k-1][0] + 1) != 0 )
                m_P[k-1][ m_dim*getFineNode(i, m_N[k-1], m_dim) + j - m_dim ][m_dim*i+j] += 0.5;

            // south node
            if ( i % ((m_N[k-1][0]+1)*(m_N[k-1][1]+1)) >= m_N[k-1][0] + 1)
                m_P[k-1][ m_dim*getFineNode(i, m_N[k-1], m_dim) + j - (m_dim*( 2*(m_N[k][0] + 1) ))/2][m_dim*i+j] += 0.5;

            // north-east node
            if ( i % ((m_N[k-1][0]+1)*(m_N[k-1][1]+1)) < (m_N[k-1][0] + 1)*(m_N[k-1][1]) && i % (m_N[k-1][0] + 1) != m_N[k-1][0] )
                m_P[k-1][ m_dim*getFineNode(i, m_N[k-1], m_dim) + j + (m_dim*( 2*(m_N[k][0] + 1) ))/2 + m_dim ][m_dim*i+j] = 0.25;

            // north-west node
            if ( i % ((m_N[k-1][0]+1)*(m_N[k-1][1]+1)) < (m_N[k-1][0] + 1)*(m_N[k-1][1]) && i % (m_N[k-1][0] + 1) != 0 )
                m_P[k-1][ m_dim*getFineNode(i, m_N[k-1], m_dim) + j + (m_dim*( 2*(m_N[k][0] + 1) ))/2 - m_dim ][m_dim*i+j] += 0.25;

            // south-east node
            if ( i % ((m_N[k-1][0]+1)*(m_N[k-1][1]+1)) >= m_N[k-1][0] + 1 && i % (m_N[k-1][0] + 1) != m_N[k-1][0] )
                m_P[k-1][ m_dim*getFineNode(i, m_N[k-1], m_dim) + j - (m_dim*( 2*(m_N[k][0] + 1) ))/2 + m_dim ][m_dim*i+j] += 0.25;

            // south-west node
            if ( i % ((m_N[k-1][0]+1)*(m_N[k-1][1]+1)) >= m_N[k-1][0] + 1 && i % (m_N[k-1][0] + 1) != 0 )
                m_P[k-1][ m_dim*getFineNode(i, m_N[k-1], m_dim) + j - (m_dim*( 2*(m_N[k][0] + 1) ))/2 - m_dim ][m_dim*i+j] += 0.25;

            if ( m_dim == 3 )
            {
                //// previous layer
                if ( i >= (m_N[k-1][0]+1)*(m_N[k-1][1]+1) )
                    m_P[k-1][ m_dim*getFineNode(i, m_N[k-1], m_dim) + j - m_dim*(m_N[k][0]+1)*(m_N[k][1]+1) ][m_dim*i+j] += 0.5;

                // previous layer-east node
                if ( i >= (m_N[k-1][0]+1)*(m_N[k-1][1]+1) && i % (m_N[k-1][0] + 1) != m_N[k-1][0] )
                    m_P[k-1][ m_dim*getFineNode(i, m_N[k-1], m_dim) + j - m_dim*(m_N[k][0]+1)*(m_N[k][1]+1) + m_dim ][m_dim*i+j] += 0.25;

                // previous layer-north node
                if ( i >= (m_N[k-1][0]+1)*(m_N[k-1][1]+1) && i % ((m_N[k-1][0]+1)*(m_N[k-1][1]+1)) < (m_N[k-1][0] + 1)*(m_N[k-1][1]) )
                    m_P[k-1][ m_dim*getFineNode(i, m_N[k-1], m_dim) + j - m_dim*(m_N[k][0]+1)*(m_N[k][1]+1) + (m_dim*( 2*(m_N[k][0] + 1) ))/2 ][m_dim*i+j] += 0.25;

                // previous layer-west node
                if ( i >= (m_N[k-1][0]+1)*(m_N[k-1][1]+1) && i % (m_N[k-1][0] + 1) != 0 )
                    m_P[k-1][ m_dim*getFineNode(i, m_N[k-1], m_dim) + j - m_dim*(m_N[k][0]+1)*(m_N[k][1]+1) - m_dim ][m_dim*i+j] += 0.25;

                // previous layer-south node
                if ( i >= (m_N[k-1][0]+1)*(m_N[k-1][1]+1) && i % ((m_N[k-1][0]+1)*(m_N[k-1][1]+1)) >= m_N[k-1][0] + 1)
                    m_P[k-1][ m_dim*getFineNode(i, m_N[k-1], m_dim) + j - m_dim*(m_N[k][0]+1)*(m_N[k][1]+1) - (m_dim*( 2*(m_N[k][0] + 1) ))/2 ][m_dim*i+j] += 0.25;

                // previous-north-east node
                if ( i >= (m_N[k-1][0]+1)*(m_N[k-1][1]+1) && i % ((m_N[k-1][0]+1)*(m_N[k-1][1]+1)) < (m_N[k-1][0] + 1)*(m_N[k-1][1]) && i % (m_N[k-1][0] + 1) != m_N[k-1][0] )
                    m_P[k-1][ m_dim*getFineNode(i, m_N[k-1], m_dim) + j - m_dim*(m_N[k][0]+1)*(m_N[k][1]+1) + (m_dim*( 2*(m_N[k][0] + 1) ))/2 + m_dim ][m_dim*i+j] += 0.125;

                // previous-north-west node
                if ( i >= (m_N[k-1][0]+1)*(m_N[k-1][1]+1) && i % ((m_N[k-1][0]+1)*(m_N[k-1][1]+1)) < (m_N[k-1][0] + 1)*(m_N[k-1][1]) && i % (m_N[k-1][0] + 1) != 0 )
                    m_P[k-1][ m_dim*getFineNode(i, m_N[k-1], m_dim) + j - m_dim*(m_N[k][0]+1)*(m_N[k][1]+1) + (m_dim*( 2*(m_N[k][0] + 1) ))/2 - m_dim ][m_dim*i+j] += 0.125;

                // previous-south-east node
                if ( i >= (m_N[k-1][0]+1)*(m_N[k-1][1]+1) && i % ((m_N[k-1][0]+1)*(m_N[k-1][1]+1)) >= m_N[k-1][0] + 1 && i % (m_N[k-1][0] + 1) != m_N[k-1][0] )
                    m_P[k-1][ m_dim*getFineNode(i, m_N[k-1], m_dim) + j - m_dim*(m_N[k][0]+1)*(m_N[k][1]+1) - (m_dim*( 2*(m_N[k][0] + 1) ))/2 + m_dim ][m_dim*i+j] += 0.125;

                // previous-south-west node
                if ( i >= (m_N[k-1][0]+1)*(m_N[k-1][1]+1) && i % ((m_N[k-1][0]+1)*(m_N[k-1][1]+1)) >= m_N[k-1][0] + 1 && i % (m_N[k-1][0] + 1) != 0)
                    m_P[k-1][ m_dim*getFineNode(i, m_N[k-1], m_dim) + j - m_dim*(m_N[k][0]+1)*(m_N[k][1]+1) - (m_dim*( 2*(m_N[k][0] + 1) ))/2 - m_dim ][m_dim*i+j] += 0.125;


                //// next layer
                if ( i < (m_N[k-1][0]+1)*(m_N[k-1][1]+1)*(m_N[k-1][2]) )
                    m_P[k-1][ m_dim*getFineNode(i, m_N[k-1], m_dim) + j + m_dim*(m_N[k][0]+1)*(m_N[k][1]+1) ][m_dim*i+j] += 0.5;

                // next layer-east node
                if ( i < (m_N[k-1][0]+1)*(m_N[k-1][1]+1)*(m_N[k-1][2]) && i % (m_N[k-1][0] + 1) != m_N[k-1][0] )
                    m_P[k-1][ m_dim*getFineNode(i, m_N[k-1], m_dim) + j + m_dim*(m_N[k][0]+1)*(m_N[k][1]+1) + m_dim ][m_dim*i+j] += 0.25;

                // next layer-north node
                if ( i < (m_N[k-1][0]+1)*(m_N[k-1][1]+1)*(m_N[k-1][2]) && i % ((m_N[k-1][0]+1)*(m_N[k-1][1]+1)) < (m_N[k-1][0] + 1)*(m_N[k-1][1]) )
                    m_P[k-1][ m_dim*getFineNode(i, m_N[k-1], m_dim) + j + m_dim*(m_N[k][0]+1)*(m_N[k][1]+1) + (m_dim*( 2*(m_N[k][0] + 1) ))/2 ][m_dim*i+j] += 0.25;

                // next layer-west node
                if ( i < (m_N[k-1][0]+1)*(m_N[k-1][1]+1)*(m_N[k-1][2]) && i % (m_N[k-1][0] + 1) != 0 )
                    m_P[k-1][ m_dim*getFineNode(i, m_N[k-1], m_dim) + j + m_dim*(m_N[k][0]+1)*(m_N[k][1]+1) - m_dim ][m_dim*i+j] += 0.25;

                // next layer-south node
                if ( i < (m_N[k-1][0]+1)*(m_N[k-1][1]+1)*(m_N[k-1][2]) && i % ((m_N[k-1][0]+1)*(m_N[k-1][1]+1)) >= m_N[k-1][0] + 1)
                    m_P[k-1][ m_dim*getFineNode(i, m_N[k-1], m_dim) + j + m_dim*(m_N[k][0]+1)*(m_N[k][1]+1) - (m_dim*( 2*(m_N[k][0] + 1) ))/2][m_dim*i+j] += 0.25;

                // next-north-east node
                if ( i < (m_N[k-1][0]+1)*(m_N[k-1][1]+1)*(m_N[k-1][2]) && i % ((m_N[k-1][0]+1)*(m_N[k-1][1]+1)) < (m_N[k-1][0] + 1)*(m_N[k-1][1]) && i % (m_N[k-1][0] + 1) != m_N[k-1][0] )
                    m_P[k-1][ m_dim*getFineNode(i, m_N[k-1], m_dim) + j + m_dim*(m_N[k][0]+1)*(m_N[k][1]+1) + (m_dim*( 2*(m_N[k][0] + 1) ))/2 + m_dim ][m_dim*i+j] += 0.125;

                // next-north-west node
                if ( i < (m_N[k-1][0]+1)*(m_N[k-1][1]+1)*(m_N[k-1][2]) && i % ((m_N[k-1][0]+1)*(m_N[k-1][1]+1)) < (m_N[k-1][0] + 1)*(m_N[k-1][1]) && i % (m_N[k-1][0] + 1) != 0 )
                    m_P[k-1][ m_dim*getFineNode(i, m_N[k-1], m_dim) + j + m_dim*(m_N[k][0]+1)*(m_N[k][1]+1) + (m_dim*( 2*(m_N[k][0] + 1) ))/2 - m_dim ][m_dim*i+j] += 0.125;

                // next-south-east node
                if ( i < (m_N[k-1][0]+1)*(m_N[k-1][1]+1)*(m_N[k-1][2]) && i % ((m_N[k-1][0]+1)*(m_N[k-1][1]+1)) >= m_N[k-1][0] + 1 && i % (m_N[k-1][0] + 1) != m_N[k-1][0] )
                    m_P[k-1][ m_dim*getFineNode(i, m_N[k-1], m_dim) + j + m_dim*(m_N[k][0]+1)*(m_N[k][1]+1) - (m_dim*( 2*(m_N[k][0] + 1) ))/2 + m_dim ][m_dim*i+j] += 0.125;

                // next-south-west node
                if ( i < (m_N[k-1][0]+1)*(m_N[k-1][1]+1)*(m_N[k-1][2]) && i % ((m_N[k-1][0]+1)*(m_N[k-1][1]+1)) >= m_N[k-1][0] + 1 && i % (m_N[k-1][0] + 1) != 0)
                    m_P[k-1][ m_dim*getFineNode(i, m_N[k-1], m_dim) + j + m_dim*(m_N[k][0]+1)*(m_N[k][1]+1) - (m_dim*( 2*(m_N[k][0] + 1) ))/2 - m_dim ][m_dim*i+j] += 0.125;

            }


            }
        }
    }
  
    
    // if ( m_dim == 3 )
    // {
        
        // DEBUG:CHECK: have to loop through the fine DOFs?
        // applying BC to relevant DOFs
        for ( int k = lev ; k != 0 ; k-- )
        {
            // loop through each element in bc_index vector
            for ( size_t bc = 0 ; bc < m_bc_index[k-1].size(); ++bc )
            {
                size_t j = m_bc_index[k-1][bc];
                size_t node_index = j / m_dim;

                // clear columns of the bc indices 
                for( int i = 0 ; i < m_numNodes[k]*m_dim ; i++ )
                        m_P[k-1][i][j] = 0;
                        

                m_P[k-1][ getFineNode(node_index, m_N[k-1], m_dim)*m_dim + (j%m_dim) ][m_dim*node_index + (j%m_dim)] = 1;
                   
            }
        }
    // }


    return true;
}

bool Assembler::assembleRestMatrix(size_t lev)
{

    for ( int k = lev ; k != 0 ; k-- )
        for ( int i = 0 ; i < m_numNodes[k-1]*m_dim ; ++i )
        {
            for ( int j = 0 ; j < m_numNodes[k]*m_dim ; ++j )
                m_R[k-1][i][j] = m_P[k-1][j][i];
        }

    return true;
}
bool Assembler::assembleRestMatrix_GPU(vector<double*> &d_r_value, vector<size_t*> &d_r_index, size_t lev)
{
    // resizing the restriction matrices according to the number of grid-levels
    d_r_value.resize( m_numLevels - 1 );
    d_r_index.resize( m_numLevels - 1 );

    // obtaining the max_row_size
    cout << m_r_max_row_size[m_topLev] << endl;
    // TODO: fix member variables


    return true;
}


// to produce an ELLmatrix of the global stiffness in the device
// will return d_value, d_index, d_max_row_size
bool Assembler::assembleGlobal(vector<size_t> &num_rows, vector<size_t> &max_row_size, vector<size_t> &p_max_row_size, vector<size_t> &r_max_row_size)
{

    // TODO: if no BC is set, return false with error
    // cout << "assembleGlobal" << endl;


    // adding nodes and elements to the top-level global grid
    for ( int i = 0 ; i < m_numNodes[m_topLev] ; ++i )
        m_node.push_back(Node(i));

    for ( int i = 0 ; i < m_numElements[m_topLev] ; ++i )
        m_element.push_back(Element(i));

    size_t numNodesIn2D = (m_N[m_topLev][0]+1)*(m_N[m_topLev][1]+1);


    // assign the nodes to each element
    if ( m_dim == 2)
    {
        for ( int i = 0 ; i < m_numElements[m_topLev] ; i++ )
        {
            m_element[i].addNode(&m_node[ i + i/m_N[m_topLev][0] ]);   // lower left node
            m_element[i].addNode(&m_node[ i + i/m_N[m_topLev][0] + 1]);   // lower right node
            m_element[i].addNode(&m_node[ i + i/m_N[m_topLev][0] + m_N[m_topLev][0] + 1]);   // upper left node
            m_element[i].addNode(&m_node[ i + i/m_N[m_topLev][0] + m_N[m_topLev][0] + 2]);   // upper right node
        }
    }

    // m_dim == 3
    else    
    {
        
        size_t numElementsIn2D = (m_N[m_topLev][0])*(m_N[m_topLev][1]);

        // loop through each element
        for ( int i = 0 ; i < m_numElements[m_topLev] ; i++ )
        {
            // base counter
            size_t base = i%numElementsIn2D;

            m_element[i].addNode(&m_node[ base + (i/numElementsIn2D)*numNodesIn2D + base/m_N[m_topLev][0] ]);   // lower left node
            m_element[i].addNode(&m_node[ base + (i/numElementsIn2D)*numNodesIn2D + base/m_N[m_topLev][0] + 1]);   // lower right node
            m_element[i].addNode(&m_node[ base + (i/numElementsIn2D)*numNodesIn2D + base/m_N[m_topLev][0] + m_N[m_topLev][0] + 1]);   // upper left node
            m_element[i].addNode(&m_node[ base + (i/numElementsIn2D)*numNodesIn2D + base/m_N[m_topLev][0] + m_N[m_topLev][0] + 2]);   // upper right node

            m_element[i].addNode(&m_node[ base + (i/numElementsIn2D)*numNodesIn2D + base/m_N[m_topLev][0] + numNodesIn2D]);
            m_element[i].addNode(&m_node[ base + (i/numElementsIn2D)*numNodesIn2D + base/m_N[m_topLev][0] + numNodesIn2D + 1]);
            m_element[i].addNode(&m_node[ base + (i/numElementsIn2D)*numNodesIn2D + base/m_N[m_topLev][0] + numNodesIn2D + m_N[m_topLev][0] + 1]);
            m_element[i].addNode(&m_node[ base + (i/numElementsIn2D)*numNodesIn2D + base/m_N[m_topLev][0] + numNodesIn2D + m_N[m_topLev][0] + 2]);

        }
        
    }

    
    
    // // // DEBUG:
    // //     for ( int elem = 0 ; elem < m_numElements[m_topLev] ; elem++ )
    // //     {
    // //         cout << "Element " << elem << endl;
    // //         for ( int i = 0 ; i < 4 ; ++i )
    // //         {
    // //             cout << m_element[elem].nodeIndex(i) << endl;
    // //         }
    // //         cout << "\n";
    // //     }
               
    

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
    // A_global = sum (A_local(GP[]) * initial_chi^p )

    if ( m_dim == 2 )
    {
        for ( int elmn_index = 0 ; elmn_index < m_numElements[m_topLev] ; elmn_index++ )
        {
            for ( int x = 0 ; x < 4 ; x++ )
            {
                for ( int y = 0 ; y < 4 ; y++ )
                {      
                        m_A_g[m_topLev][ 2*m_element[elmn_index].nodeIndex(x)     ][ 2*m_element[elmn_index].nodeIndex(y)     ] += pow(m_rho, m_p) * valueAt( 2*x    , 2*y     );
                        m_A_g[m_topLev][ 2*m_element[elmn_index].nodeIndex(x)     ][ 2*m_element[elmn_index].nodeIndex(y) + 1 ] += pow(m_rho, m_p) * valueAt( 2*x    , 2*y + 1 );
                        m_A_g[m_topLev][ 2*m_element[elmn_index].nodeIndex(x) + 1 ][ 2*m_element[elmn_index].nodeIndex(y)     ] += pow(m_rho, m_p) * valueAt( 2*x + 1, 2*y     );
                        m_A_g[m_topLev][ 2*m_element[elmn_index].nodeIndex(x) + 1 ][ 2*m_element[elmn_index].nodeIndex(y) + 1 ] += pow(m_rho, m_p) * valueAt( 2*x + 1, 2*y + 1 );
                }
            }
        }
    }

    if ( m_dim == 3 )
    {
        for ( int elmn_index = 0 ; elmn_index < m_numElements[m_topLev] ; elmn_index++ )
        {

            for ( int x = 0 ; x < 8 ; x++ )
            {
                for ( int y = 0 ; y < 8 ; y++ )
                {      
                            m_A_g[m_topLev][ 3*m_element[elmn_index].nodeIndex(x)     ][ 3*m_element[elmn_index].nodeIndex(y)     ] += pow(m_rho, m_p) * valueAt( 3*x    , 3*y     );
                            m_A_g[m_topLev][ 3*m_element[elmn_index].nodeIndex(x)     ][ 3*m_element[elmn_index].nodeIndex(y) + 1 ] += pow(m_rho, m_p) * valueAt( 3*x    , 3*y + 1 );
                            m_A_g[m_topLev][ 3*m_element[elmn_index].nodeIndex(x)     ][ 3*m_element[elmn_index].nodeIndex(y) + 2 ] += pow(m_rho, m_p) * valueAt( 3*x    , 3*y + 2 );
                            m_A_g[m_topLev][ 3*m_element[elmn_index].nodeIndex(x) + 1 ][ 3*m_element[elmn_index].nodeIndex(y)     ] += pow(m_rho, m_p) * valueAt( 3*x + 1, 3*y     );
                            m_A_g[m_topLev][ 3*m_element[elmn_index].nodeIndex(x) + 1 ][ 3*m_element[elmn_index].nodeIndex(y) + 1 ] += pow(m_rho, m_p) * valueAt( 3*x + 1, 3*y + 1 );
                            m_A_g[m_topLev][ 3*m_element[elmn_index].nodeIndex(x) + 1 ][ 3*m_element[elmn_index].nodeIndex(y) + 2 ] += pow(m_rho, m_p) * valueAt( 3*x + 1, 3*y + 2 );
                            m_A_g[m_topLev][ 3*m_element[elmn_index].nodeIndex(x) + 2 ][ 3*m_element[elmn_index].nodeIndex(y)     ] += pow(m_rho, m_p) * valueAt( 3*x + 2, 3*y     );
                            m_A_g[m_topLev][ 3*m_element[elmn_index].nodeIndex(x) + 2 ][ 3*m_element[elmn_index].nodeIndex(y) + 1 ] += pow(m_rho, m_p) * valueAt( 3*x + 2, 3*y + 1 );
                            m_A_g[m_topLev][ 3*m_element[elmn_index].nodeIndex(x) + 2 ][ 3*m_element[elmn_index].nodeIndex(y) + 2 ] += pow(m_rho, m_p) * valueAt( 3*x + 2, 3*y + 2 );
                }
            }
        }
    }


    // //DEBUG:
    // for ( int x = 0 ; x < m_numNodes[m_topLev]*m_dim ; x++ )
    // {
    //     for ( int y = 0 ; y < m_numNodes[m_topLev]*m_dim ; y++ )
    //     {      
    //             cout << m_A_g[m_topLev][x][y] << " ";
    //     }

    //     cout << endl;
    // }


    // // cleanup: replacing any values <1e-7 to 0.0
    // for ( int x = 0 ; x < m_numNodes[m_topLev]*m_dim ; x++ )
    // {
    //     for ( int y = 0 ; y < m_numNodes[m_topLev]*m_dim ; y++ )
    //     {      
    //         if ( m_A_g[m_topLev][x][y] < 1e-7 && m_A_g[m_topLev][x][y] > -1e-7)
    //             m_A_g[m_topLev][x][y] = 0.0;
    //     }
    // }

    // // DEBUG:
    // for ( int x = 0 ; x < m_numNodes[m_topLev]*m_dim ; x++ )
    // {
    //     for ( int y = 0 ; y < m_numNodes[m_topLev]*m_dim ; y++ )
    //     {      
    //             cout << m_A_g[m_topLev][x][y] << " ";
    //     }

    //     cout << endl;
    // }



    // applying BC on the matrix
    // DOFs which are affected by BC will have identity rows/cols { 0 0 .. 1 .. 0 0}
    for ( int i = 0 ; i < m_bc_index[m_topLev].size() ; ++i )
        applyMatrixBC(m_A_g[m_topLev], m_bc_index[m_topLev][i], num_rows[m_topLev], m_dim);


    // // DEBUG:
    // for ( int i = 0 ; i < num_rows[m_topLev] ; i++ )
    // {
    //     for ( int j = 0 ; j < num_rows[m_topLev] ; j++ )
    //         cout << m_A_g[m_topLev][i][j] << " ";

    //     cout << "\n";
    // }

    
    


    // filling in the coarse matrices of each level
    for ( int lev = m_topLev ; lev != 0 ; lev-- )
        PTAP(m_A_g[lev-1], m_A_g[lev], m_P[lev-1], num_rows[lev], num_rows[lev-1] );

    

    

    // // DEBUG:
    // int bbbb = 1;
    // for ( int i = 0 ; i < num_rows[bbbb] ; i++ )
    // {
    //     for ( int j = 0 ; j < num_rows[bbbb] ; j++ )
    //         cout << m_A_g[bbbb][i][j] << " ";

    //     cout << "\n";
    // }





    //// obtaining the ELLPACK value and index vectors from the global stiffness matrix

    // resizing the vectors required for ELLPACK for each level
    max_row_size.resize(m_numLevels);
    p_max_row_size.resize(m_numLevels - 1);
    r_max_row_size.resize(m_numLevels - 1);

    // calculate global max_num_rows, which will also be needed when allocating memory in device
    for ( int lev = 0 ; lev < m_numLevels ; lev++ )
        max_row_size[lev] = getMaxRowSize(m_A_g[lev], num_rows[lev], num_rows[lev]);
    
    for ( int lev = 0 ; lev < m_numLevels - 1 ; lev++ )
        p_max_row_size[lev] = getMaxRowSize(m_P[lev], num_rows[lev+1], num_rows[lev]);
    
    for ( int lev = 0 ; lev < m_numLevels - 1 ; lev++ )
        r_max_row_size[lev] = getMaxRowSize(m_R[lev], num_rows[lev], num_rows[lev+1]);
    

    // resizing the vectors
    m_p_value_g.resize( m_numLevels - 1 );
    m_p_index_g.resize( m_numLevels - 1 );
    m_r_value_g.resize( m_numLevels - 1 );
    m_r_index_g.resize( m_numLevels - 1 );
    m_value_g.resize( m_numLevels );
    m_index_g.resize( m_numLevels );

    // prolongation matrices
    for ( int lev = 0 ; lev < m_numLevels - 1 ; lev++ )
        transformToELL(m_P[lev], m_p_value_g[lev], m_p_index_g[lev], p_max_row_size[lev], num_rows[lev+1], num_rows[lev] );    

    // restriction matrices
    for ( int lev = 0 ; lev < m_numLevels - 1 ; lev++ )
        transformToELL(m_R[lev], m_r_value_g[lev], m_r_index_g[lev], r_max_row_size[lev], num_rows[lev], num_rows[lev+1] );    

    // stiffness matrices
    for ( int lev = 0 ; lev < m_numLevels ; lev++ )
        transformToELL(m_A_g[lev], m_value_g[lev], m_index_g[lev], max_row_size[lev], num_rows[lev], num_rows[lev] );



    // int a = 0;
    // for ( int j = 0 ; j < num_rows[1] ; j++ )
    // {
    //     for ( int i = 0 ; i < max_row_size[1] ; i++ )
    //     {
    //         cout << m_value_g[1][a] << " ";
    //         a++;
    //     }

    //     cout << "\n";
    // }

    // int a = 0;
    // for ( int j = 0 ; j < num_rows[1] ; j++ )
    // {
    //     for ( int i = 0 ; i < max_row_size[1] ; i++ )
    //     {
    //         cout << m_index_g[1][a] << " ";
    //         a++;
    //     }

    //     cout << "\n";
    // }



    // // NOTE: can somehow do init for solving now while allocating memory in device?
    // // do async malloc then your init() should be AFTER the memcpy stuff, not before

    return true;

}

bool Assembler::UpdateGlobalStiffness(
    double* &d_chi, 
    vector<double*> &d_value, vector<size_t*> &d_index,         // global stiffness
    vector<double*> &d_p_value, vector<size_t*> &d_p_index,     // prolongation matrices
    vector<double*> &d_r_value, vector<size_t*> &d_r_index,     // restriction matrices
    double* &d_A_local)                                         // local stiffness matrix
{
    
    // reinitialize relevant variables
    // stiffness matrices, A
    for ( int lev = 0 ; lev < m_numLevels ; ++lev )
        setToZero<<<m_ell_gridDim[lev], m_ell_blockDim[lev]>>>( d_value[lev], m_num_rows[lev]*m_max_row_size[lev]);


    // // printELLrow(2, d_value[2], d_index[2], m_max_row_size[2], m_num_rows[2], m_num_rows[2]);
    // // printELLrow(1, d_value[1], d_index[1], m_max_row_size[1], m_num_rows[1], m_num_rows[1]);
    
    dim3 l_blockDim(m_num_rows_l,m_num_rows_l,1);

    // printVector_GPU<<<1,4>>>( d_chi, 4);


    // assemble the global stiffness matrix on the finest grid with the updated chi of each element
    for ( int i = 0 ; i < m_numElements[m_topLev] ; ++i )
        assembleGrid2D_GPU<<<1,l_blockDim>>>( m_N[m_topLev][0], m_dim, &d_chi[i], d_A_local, &d_value[m_topLev][0], &d_index[m_topLev][0], m_max_row_size[m_topLev], m_num_rows_l, m_d_node_index[i], m_p);

    
    // printELLrow(2, d_value[2], d_index[2], m_max_row_size[2], m_num_rows[2], m_num_rows[2]);

    // calculating the needed cuda 2D grid size for the global assembly
    dim3 g_gridDim;
    dim3 g_blockDim;

    // for ( int i = 0 ; i < m_bc_index[m_topLev].size() ; i++ )
    // cout << m_bc_index[m_topLev][i] << endl;
    
    // TODO: CHECK: this is a bit shaky
    // TODO: think it's a bit overkill to use a lot of cuda threads here
    //// apply boundary condition to global and P/R matrices
    // global stiffness matrix
    calculateDimensions2D( m_num_rows[m_topLev], m_num_rows[m_topLev], g_gridDim, g_blockDim);
    for ( int i = 0 ; i < m_bc_index[m_topLev].size() ; i++ )
        applyMatrixBC_GPU_test<<<g_gridDim,g_blockDim>>>(&d_value[m_topLev][0], &d_index[m_topLev][0], m_max_row_size[m_topLev], m_bc_index[m_topLev][i], m_num_rows[m_topLev], m_num_rows[m_topLev] );



    // printELLrow(2, d_value[2], d_index[2], m_max_row_size[2], m_num_rows[2], m_num_rows[2]);


    // // TODO: use optimized matrix multiplication
    dim3 temp_gridDim;
    dim3 temp_blockDim;
       
    // // d_temp_matrix[lev-1][lev] to store R*A
    // double* d_temp_matrix;
    // CUDA_CALL( cudaMalloc((void**)&d_temp_matrix, sizeof(double) * num_rows[m_topLev] * num_rows[m_topLev-1] ) );
    // CUDA_CALL( cudaMemset( d_temp_matrix, 0, sizeof(double) * num_rows[m_topLev] * num_rows[m_topLev-1] ) );
    

    // A_coarse = R * A_fine * P
    for ( int lev = m_topLev ; lev != 0 ; lev--)
    {
        calculateDimensions(m_num_rows[lev] * m_num_rows[lev-1], temp_gridDim, temp_blockDim);
        setToZero<<<temp_gridDim, temp_blockDim>>>( m_d_temp_matrix, m_num_rows[lev] * m_num_rows[lev-1]);
        RAP( d_value, d_index, m_max_row_size, d_r_value, d_r_index, m_r_max_row_size, d_p_value, d_p_index, m_p_max_row_size, m_d_temp_matrix, m_num_rows, lev);
    }






    // // // // DEBUG: temp :
    // // // vector<vector<size_t>> temp_bc_index(2);

    // // // temp_bc_index[0] = {0,1 ,4,5};
    // // // temp_bc_index[1] = {0,1 ,6,7, 12,13};
    
    // // // DEBUG: temp: not optimized
    // // // d_temp_matrix[8][18] to store R*A
    // double* d_temp_matrix;
    // CUDA_CALL( cudaMalloc((void**)&d_temp_matrix, sizeof(double) * m_num_rows[m_topLev] * m_num_rows[m_topLev-1] ) );
    // CUDA_CALL( cudaMemset( d_temp_matrix, 0, sizeof(double) * m_num_rows[m_topLev] * m_num_rows[m_topLev-1] ) );
    
    // // calculating the needed cuda 2D grid size for the global assembly
    // dim3 g_gridDim;
    // dim3 g_blockDim;
    // calculateDimensions2D( m_num_rows[m_topLev], m_num_rows[m_topLev], g_gridDim, g_blockDim);

    

    // // applying the boundary conditions on the global stiffness matrix   
    // for ( int i = 0 ; i < m_bc_index[m_topLev].size() ; i++ )
    //     applyMatrixBC_GPU<<<g_gridDim,g_blockDim>>>(&d_value[m_topLev][0], &d_index[m_topLev][0], m_max_row_size[m_topLev], m_bc_index[m_topLev][i], m_num_rows[m_topLev] );




    // // cudaDeviceSynchronize();
    // // printELLrow(2, d_value[2], d_index[2], m_max_row_size[2], m_num_rows[2], m_num_rows[2]);



    // // // TODO: use optimized matrix multiplication
    // dim3 temp_gridDim;
    // dim3 temp_blockDim;
       
    // // A_coarse = R * A_fine * P
    // for ( int lev = m_topLev ; lev != 0 ; lev--)
    // {
    //     calculateDimensions(m_num_rows[lev] * m_num_rows[lev-1], temp_gridDim, temp_blockDim);
    //     setToZero<<<temp_gridDim, temp_blockDim>>>( d_temp_matrix, m_num_rows[lev] * m_num_rows[lev-1]);
    //     RAP( d_value, d_index, m_max_row_size, d_r_value, d_r_index, m_r_max_row_size, d_p_value, d_p_index, m_p_max_row_size, d_temp_matrix, m_num_rows, lev);
    // }
    // // // // RAP( d_value, d_index, m_max_row_size, d_r_value, d_r_index, m_r_max_row_size, d_p_value, d_p_index, m_p_max_row_size, d_temp_matrix, m_num_rows, m_topLev-1);

    //     // setToZero<<<1,m_num_rows[m_topLev] * m_num_rows[m_topLev-1]>>>( d_temp_matrix, m_num_rows[m_topLev] * m_num_rows[m_topLev-1]);
    // // // 	printVector_GPU<<<1,144>>>( d_temp_matrix, 144 );




    // cudaDeviceSynchronize();
    // // printELLrow(1, d_p_value[1], d_p_index[1], m_p_max_row_size[1], m_num_rows[2], m_num_rows[1]);
    // // printELLrow(1, d_value[1], d_index[1], m_max_row_size[1], m_num_rows[1], m_num_rows[1]);
    // // printELLrow(0, d_value[0], d_index[0], m_max_row_size[0], m_num_rows[0], m_num_rows[0]);

    // printELLrow(0, d_value[0], d_index[0], max_row_size[0], num_rows[0], num_rows[0]);
    // printELLrow(1, d_value[1], d_index[1], max_row_size[1], num_rows[1], num_rows[1]);
    // printELLrow(2, d_value[2], d_index[2], max_row_size[2], num_rows[2], num_rows[2]);
    // printELLrow(0, d_r_value[0], d_r_index[0], r_max_row_size[0], num_rows[0], num_rows[1]);
    // printELLrow(1, d_r_value[1], d_r_index[1], r_max_row_size[1], num_rows[1], num_rows[2]);
    // printELLrow(0, d_p_value[0], d_p_index[0], p_max_row_size[0], num_rows[1], num_rows[0]);
    // printELLrow(1, d_p_value[1], d_p_index[1], p_max_row_size[1], num_rows[2], num_rows[1]);

    // printVector_GPU<<<1,10>>>( dt_index, 10 );
    // printLinearVector( d_index[0], num_rows[0], max_row_size[0]);
    // printLinearVector( d_index[1], num_rows[1], max_row_size[1]);
    // printLinearVector( d_index[2], num_rows[2], max_row_size[2]);
    // printLinearVector( d_A_local, 8, 8);
    // printLinearVector( d_temp_matrix, 16, 42);

	cudaDeviceSynchronize();


    return true;
}

