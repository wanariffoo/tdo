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
                max_row_size[lev] = 18;

            // loop through each level
            for (int lev = m_topLev - 1; lev != 0 ; --lev)
            {
                p_max_row_size[lev] = 4;
                r_max_row_size[lev] = 9;
            }
        }

        // dim = 3
        else 
        {
            if (m_N[0][0] == 1 && m_N[0][1] == 1 && m_N[0][2] == 1)
            {
                max_row_size[0] = 24;
                p_max_row_size[0] = 8; // CHECK: not sure
                r_max_row_size[0] = 8;
                
            }

            else if (m_N[0][0] == 3 && m_N[0][1] == 1 && m_N[0][2] == 1 )
            {
                max_row_size[0] = 36;
                p_max_row_size[0] = 8;
                r_max_row_size[0] = 12;
            }

            else
            {   
                throw(runtime_error("TODO: not done"));
            }


            // TODO: CHECK: is the number correct?
            // loop through each level (stiffness matrix)
            for (int lev = m_topLev ; lev != 0 ; --lev)
                max_row_size[lev] = 27*3;
            
            // loop through each level (prolongation & restriction matrices)
            for (int lev = m_topLev - 1; lev != 0 ; --lev)
            {
                p_max_row_size[lev] = 8;
                r_max_row_size[lev] = 27;
            }
            
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

    

    // TODO: 3D doesn't work yet
    // assembling the local stiffness matrix
    assembleLocal();
 
    //// assembling prolongation & restriction matrices

    // TODO: assemble in GPU, currently it's assembling in CPU
    cout << "Assembling prol matrix ..." << endl;
    assembleProlMatrix_GPU(d_p_value, d_p_index, m_topLev);

    cout << "Assembling rest matrix ..." << endl;
    assembleRestMatrix_GPU(d_r_value, d_r_index, d_p_value, d_p_index);

//     // allocating temp matrix to be used in RAP
//     CUDA_CALL( cudaMalloc((void**)&m_d_temp_matrix, sizeof(double) * num_rows[m_topLev] * num_rows[m_topLev-1] ) );
//     CUDA_CALL( cudaMemset( m_d_temp_matrix, 0, sizeof(double) * num_rows[m_topLev] * num_rows[m_topLev-1] ) );
    


//     //// adding nodes and elements to the top-level global grid
//     for ( int i = 0 ; i < m_numNodes[m_topLev] ; ++i )
//         m_node.push_back(Node(i));

//     for ( int i = 0 ; i < m_numElements[m_topLev] ; ++i )
//         m_element.push_back(Element(i));

//     size_t numNodesIn2D = (m_N[m_topLev][0]+1)*(m_N[m_topLev][1]+1);


//     // assigning the nodes to each element
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
//         for ( int i = 0 ; i < m_numElements[m_topLev] ; i++ )
//         {
//             size_t elemcount_2D = (m_N[m_topLev][0])*(m_N[m_topLev][1]); 
//             size_t gridsize_2D = (m_N[m_topLev][0]+1)*(m_N[m_topLev][1]+1);
//             size_t multiplier = i / elemcount_2D;
//             size_t base_id = i % elemcount_2D;

//             m_element[i].addNode(&m_node[ base_id + base_id/m_N[m_topLev][0] + multiplier*gridsize_2D ]);   // lower left node
//             m_element[i].addNode(&m_node[ base_id + base_id/m_N[m_topLev][0] + multiplier*gridsize_2D + 1]);   // lower right node
//             m_element[i].addNode(&m_node[ base_id + base_id/m_N[m_topLev][0] + multiplier*gridsize_2D + m_N[m_topLev][0] + 1]);   // upper left node
//             m_element[i].addNode(&m_node[ base_id + base_id/m_N[m_topLev][0] + multiplier*gridsize_2D + m_N[m_topLev][0] + 2]);   // upper right node
            
//             // next layer
//             m_element[i].addNode(&m_node[ base_id + base_id/m_N[m_topLev][0] + multiplier*gridsize_2D + gridsize_2D]);   // lower left node
//             m_element[i].addNode(&m_node[ base_id + base_id/m_N[m_topLev][0] + multiplier*gridsize_2D + 1 + gridsize_2D]);   // lower right node
//             m_element[i].addNode(&m_node[ base_id + base_id/m_N[m_topLev][0] + multiplier*gridsize_2D + m_N[m_topLev][0] + 1 + gridsize_2D]);   // upper left node
//             m_element[i].addNode(&m_node[ base_id + base_id/m_N[m_topLev][0] + multiplier*gridsize_2D + m_N[m_topLev][0] + 2 + gridsize_2D]);   // upper right node

//         }
//     }

//     // // DEBUG:
//     //     for ( int elem = 0 ; elem < m_numElements[m_topLev] ; elem++ )
//     //     {
//     //         cout << "Element " << elem << " ";
//     //         for ( int i = 0 ; i < 8 ; ++i )
//     //         {
//     //             cout << m_element[elem].nodeIndex(i) << " ";
//     //         }
//     //         cout << "\n";
//     //     }


//     m_node_index.resize(m_numElements[m_topLev]);
//     d_node_index.resize(m_numElements[m_topLev]);
//     for ( int elem = 0 ; elem < m_numElements[m_topLev] ; elem++ )
//     {
//         for ( int index = 0 ; index < pow(2, m_dim) ; index++ )
//             m_node_index[elem].push_back( m_element[elem].getNodeIndex(index) );
//     }

//     // allocating and copying the design variable to device
//     // design variable currently has initial values of rho
//     CUDA_CALL( cudaMalloc((void**)&d_chi, sizeof(double) * m_numElements[m_topLev] ) );
//     CUDA_CALL( cudaMemcpy(d_chi, &m_chi[0], sizeof(double) * m_numElements[m_topLev], cudaMemcpyHostToDevice) );

//     // allocating and copying the (linear vector) local stiffness matrix to device
//     CUDA_CALL( cudaMalloc((void**)&d_A_local, sizeof(double) * m_num_rows_l*m_num_rows_l ) );
//     CUDA_CALL( cudaMemcpy( d_A_local, &m_A_local[0], sizeof(double) * m_num_rows_l*m_num_rows_l, cudaMemcpyHostToDevice) );

    
//     // calculating the number of nodes in a local element
//     size_t numNodes_local = pow(2,m_dim);

//     // copying the node index vector to device
//     for ( int i = 0 ; i < m_numElements[m_topLev] ; i++ )
//     {
//         CUDA_CALL( cudaMalloc( (void**)&d_node_index[i], sizeof(size_t) * numNodes_local) );
//         CUDA_CALL( cudaMemcpy( d_node_index[i], &m_node_index[i][0], sizeof(size_t) * numNodes_local, cudaMemcpyHostToDevice) );
//     }

//     m_d_node_index = d_node_index;

//     //// allocating the global matrices of each level to device
//     // matrices are empty for now, will be filled in later

//     // resizing global matrices for each grid-level
//     d_value.resize( m_numLevels );
//     d_index.resize( m_numLevels );

//     for ( int lev = 0 ; lev < m_numLevels ; lev++ )
//     {
//         CUDA_CALL( cudaMalloc((void**)&d_value[lev], sizeof(double) * max_row_size[lev] * num_rows[lev] ) );
//         CUDA_CALL( cudaMemset( d_value[lev], 0, sizeof(double) * num_rows[lev]*max_row_size[lev] ) );
//         CUDA_CALL( cudaMalloc((void**)&d_index[lev], sizeof(size_t) * max_row_size[lev] * num_rows[lev] ) );
//         CUDA_CALL( cudaMemset( d_index[lev], 0, sizeof(size_t) * num_rows[lev]*max_row_size[lev] ) );
//     }
    


//     // TODO: parallelizable
//     // filling in global stiffness matrix's ELLPACK index vector for all levels
//     dim3 index_gridDim;
//     dim3 index_blockDim;

//     if ( m_dim == 2)
//     {
//         for (int lev = m_topLev ; lev >= 0 ; --lev )
//         {
//             calculateDimensions( num_rows[lev], index_gridDim, index_blockDim);
//             fillIndexVector2D_GPU<<<index_gridDim,index_blockDim>>>(d_index[lev], m_N[lev][0], m_N[lev][1], max_row_size[lev], num_rows[lev]);
//         }
//     }

//     else
//     {   
//         for (int lev = m_topLev ; lev >= 0 ; --lev )
//         {
//             calculateDimensions( num_rows[lev], index_gridDim, index_blockDim);
//             fillIndexVector3D_GPU<<<index_gridDim,index_blockDim>>>(d_index[lev], m_N[lev][0], m_N[lev][1], m_N[lev][2], max_row_size[lev], num_rows[lev]);
//         }
//     }
    
    

//     //// filling in the top level global stiffness matrix
//     // CUDA block size for assembling the global stiffness matrix
//     dim3 l_blockDim(m_num_rows_l,m_num_rows_l,1);

//     // filling in from each element
//     for ( int i = 0 ; i < m_numElements[m_topLev] ; ++i )
//         assembleGrid2D_GPU<<<1,l_blockDim>>>( m_N[m_topLev][0], m_dim, &d_chi[i], d_A_local, &d_value[m_topLev][0], &d_index[m_topLev][0], max_row_size[m_topLev], m_num_rows_l, d_node_index[i], m_p);

//     cudaDeviceSynchronize();

// // printELLrow(1, d_value[1], d_index[1], max_row_size[1], num_rows[1], num_rows[1]);

//     // calculating the needed cuda 2D grid size for the global assembly
//     dim3 g_gridDim;
//     dim3 g_blockDim;

//     // for ( int i = 0 ; i < m_bc_index[m_topLev].size() ; i++ )
//     // cout << m_bc_index[m_topLev][i] << endl;
    
//     // TODO: CHECK: this is a bit shaky
//     // TODO: think it's a bit overkill to use a lot of cuda threads here
//     //// apply boundary conditions to global stiffness matrix
//     // global stiffness matrix
//     calculateDimensions2D( num_rows[m_topLev], num_rows[m_topLev], g_gridDim, g_blockDim);
//     for ( int i = 0 ; i < m_bc_index[m_topLev].size() ; i++ )
//         applyMatrixBC_GPU_test<<<g_gridDim,g_blockDim>>>(&d_value[m_topLev][0], &d_index[m_topLev][0], max_row_size[m_topLev], m_bc_index[m_topLev][i], num_rows[m_topLev], num_rows[m_topLev] );

//         // NOTE: optional?
//     // prolongation matrix
//     // calculateDimensions2D( num_rows[m_topLev-1], num_rows[m_topLev], g_gridDim, g_blockDim);
//     // for ( int i = 0 ; i < m_bc_index[m_topLev].size() ; i++ )
//     //     applyMatrixBC_GPU_test<<<g_gridDim,g_blockDim>>>(&d_p_value[m_topLev-1][0], &d_p_index[m_topLev-1][0], p_max_row_size[m_topLev-1], m_bc_index[m_topLev][i], num_rows[m_topLev], num_rows[m_topLev-1] );




//     //// obtaining the coarse stiffness matrices of each lower grid level
//     // // TODO: use optimized matrix multiplication
//     dim3 temp_gridDim;
//     dim3 temp_blockDim;

    
//     // A_coarse = R * A_fine * P
//     for ( int lev = m_topLev ; lev != 0 ; lev--)
//     {
//         calculateDimensions2D( num_rows[lev-1], num_rows[lev-1], temp_gridDim, temp_blockDim);
//         RAP_<<<temp_gridDim,temp_blockDim>>>(   d_value[lev], d_index[lev], max_row_size[lev], num_rows[lev], 
//                                                 d_value[lev-1], d_index[lev-1], max_row_size[lev-1], num_rows[lev-1], 
//                                                 d_r_value[lev-1], d_r_index[lev-1], r_max_row_size[lev-1],
//                                                 d_p_value[lev-1], d_p_index[lev-1], p_max_row_size[lev-1], lev-1);
//         cudaDeviceSynchronize();
//     }


    
//     // cout << "max_row_size[1]" << endl;
    // printELLrow(0, d_value[0], d_index[0], max_row_size[0], num_rows[0], num_rows[0]);
    // printELLrow(1, d_value[1], d_index[1], max_row_size[1], num_rows[1], num_rows[1]);
    // printELLrow(2, d_value[2], d_index[2], max_row_size[2], num_rows[2], num_rows[2]);
    // printELLrow(0, d_r_value[0], d_r_index[0], r_max_row_size[0], num_rows[0], num_rows[1]);
//     // printELLrow(1, d_r_value[1], d_r_index[1], r_max_row_size[1], num_rows[1], num_rows[2]);
    // printELLrow(0, d_p_value[0], d_p_index[0], p_max_row_size[0], num_rows[1], num_rows[0]);
    // printELLrow(1, d_p_value[1], d_p_index[1], p_max_row_size[1], num_rows[2], num_rows[1]);

//     // printVector_GPU<<<1,10>>>( dt_index, 10 );
//     // printLinearVector( d_index[0], num_rows[0], max_row_size[0]);
    // printLinearVector( d_index[1], num_rows[1], max_row_size[1]);
//     // printLinearVector( d_index[2], num_rows[2], max_row_size[2]);
    // printLinearVector( d_p_index[0], num_rows[1], p_max_row_size[0]);
    // printLinearVector( d_A_local, 8, 8);
//     // printLinearVector( m_d_temp_matrix, num_rows[1], num_rows[2]);
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

double Assembler::valueAt(size_t row, size_t col)
{
    return m_A_local[col + row*m_num_rows_l];
}


bool Assembler::assembleProlMatrix_GPU(
    vector<double*> &d_p_value, 
    vector<size_t*> &d_p_index, 
    size_t lev)
{
    // resizing the prolongation matrices according to the number of grid-levels for cuda
    d_p_value.resize( m_numLevels - 1 );
    d_p_index.resize( m_numLevels - 1 );

    // allocating and copying the value & index vectors to device
    for ( int lev = 0 ; lev < m_numLevels - 1 ; lev++ )
    {
        CUDA_CALL( cudaMalloc((void**)&d_p_value[lev], sizeof(double) * m_p_max_row_size[lev] * m_num_rows[lev+1] ) );
        CUDA_CALL( cudaMalloc((void**)&d_p_index[lev], sizeof(size_t) * m_p_max_row_size[lev] * m_num_rows[lev+1] ) );
    }

    dim3 gridDim;
    dim3 blockDim;

    if ( m_dim == 2)
    {
        // fill in prolongation matrix's ELLPACK index vector
        for ( int lev = 0 ; lev < m_numLevels - 1 ; lev++ )
        {
            calculateDimensions(m_num_rows[lev+1], gridDim, blockDim);
            fillProlMatrix2D_GPU<<<gridDim,blockDim>>>( d_p_value[lev], d_p_index[lev], m_N[lev+1][0], m_N[lev+1][1], m_p_max_row_size[lev], m_num_rows[lev+1], m_num_rows[lev]);
        }

    }

    // m_dim = 3
    else
    {
            int lev = 0;
            calculateDimensions(m_num_rows[lev+1], gridDim, blockDim);
            fillProlMatrix3D_GPU<<<gridDim,blockDim>>>( d_p_value[lev], d_p_index[lev], m_N[lev+1][0], m_N[lev+1][1], m_N[lev+1][2], m_p_max_row_size[lev], m_num_rows[lev+1], m_num_rows[lev]);

    }
           


    // printLinearVector( d_p_index[0], m_num_rows[1], m_p_max_row_size[0]);
    // printELLrow(0, d_p_value[0], d_p_index[0], m_p_max_row_size[0], m_num_rows[1], m_num_rows[0]);

    return true;
}

bool Assembler::assembleRestMatrix_GPU(
    vector<double*> &d_r_value, 
    vector<size_t*> &d_r_index, 
    vector<double*> &d_p_value, 
    vector<size_t*> &d_p_index)
{
    // resizing the restriction matrices according to the number of grid-levels
    d_r_value.resize( m_numLevels - 1 );
    d_r_index.resize( m_numLevels - 1 );

    // memcpy to device
    for ( int lev = 0 ; lev < m_numLevels - 1 ; lev++ )
    {
        CUDA_CALL( cudaMalloc((void**)&d_r_value[lev], sizeof(double) * m_num_rows[lev] * m_r_max_row_size[lev] ) );
        CUDA_CALL( cudaMemset( d_r_value[lev], 0, sizeof(double) * m_num_rows[lev] * m_r_max_row_size[lev] ) );
        CUDA_CALL( cudaMalloc((void**)&d_r_index[lev], sizeof(size_t) * m_num_rows[lev] * m_r_max_row_size[lev] ) );
        CUDA_CALL( cudaMemset( d_r_index[lev], 0, sizeof(size_t) * m_num_rows[lev] * m_r_max_row_size[lev] ) );
    }

    dim3 gridDim;
    dim3 blockDim;

    if ( m_dim == 2)
    {
        // fill in restriction matrix's ELLPACK index vector
        for ( int lev = 0 ; lev < m_numLevels - 1 ; lev++ )
        {
            calculateDimensions(m_numNodes[lev]*m_dim, gridDim, blockDim);
            fillIndexVectorRest2D_GPU<<<gridDim,blockDim>>>(d_r_index[lev], m_N[lev][0], m_N[lev][1], m_r_max_row_size[lev], m_num_rows[lev], m_num_rows[lev+1]);
        }

        // fill in restriction matrix's values, taken from prolongation matrix
        for ( int lev = 0 ; lev < m_numLevels - 1 ; lev++ )
        {
            calculateDimensions2D(m_num_rows[lev], m_num_rows[lev+1], gridDim, blockDim);
            fillRestMatrix<<<gridDim, blockDim>>>(d_r_value[lev], d_r_index[lev], m_r_max_row_size[lev], d_p_value[lev], d_p_index[lev], m_p_max_row_size[lev], m_num_rows[lev], m_num_rows[lev+1]);
        }
    }

    else
    {
        // fill in restriction matrix's ELLPACK index vector
        for ( int lev = 0 ; lev < m_numLevels - 1 ; lev++ )
        {
            calculateDimensions(m_numNodes[lev]*m_dim, gridDim, blockDim);
            fillIndexVectorRest3D_GPU<<<gridDim,blockDim>>>(d_r_index[lev], m_N[lev][0], m_N[lev][1], m_N[lev][2], m_r_max_row_size[lev], m_num_rows[lev], m_num_rows[lev+1]);
        }

        // fill in restriction matrix's values, taken from prolongation matrix
        for ( int lev = 0 ; lev < m_numLevels - 1 ; lev++ )
        {
            calculateDimensions2D(m_num_rows[lev], m_num_rows[lev+1], gridDim, blockDim);
            fillRestMatrix<<<gridDim, blockDim>>>(d_r_value[lev], d_r_index[lev], m_r_max_row_size[lev], d_p_value[lev], d_p_index[lev], m_p_max_row_size[lev], m_num_rows[lev], m_num_rows[lev+1]);
        }


    }
    
    cudaDeviceSynchronize();

    // printLinearVector( d_r_index[0], m_num_rows[0], m_r_max_row_size[0]);
    // printLinearVector( d_r_index[1], m_num_rows[1], m_r_max_row_size[1]);
    // printLinearVector( d_r_value[0], m_num_rows[0], m_r_max_row_size[0]);
    // printELLrow(0, d_r_value[0], d_r_index[0], m_r_max_row_size[0], m_num_rows[0], m_num_rows[1]);


    
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
        calculateDimensions2D( m_num_rows[lev-1], m_num_rows[lev-1], temp_gridDim, temp_blockDim);
        RAP_<<<temp_gridDim,temp_blockDim>>>(   d_value[lev], d_index[lev], m_max_row_size[lev], m_num_rows[lev], 
                                                d_value[lev-1], d_index[lev-1], m_max_row_size[lev-1], m_num_rows[lev-1], 
                                                d_r_value[lev-1], d_r_index[lev-1], m_r_max_row_size[lev-1],
                                                d_p_value[lev-1], d_p_index[lev-1], m_p_max_row_size[lev-1], lev-1);
        cudaDeviceSynchronize();
    }






    // // A_coarse = R * A_fine * P
    // for ( int lev = m_topLev ; lev != 0 ; lev--)
    // {
    //     calculateDimensions(m_num_rows[lev] * m_num_rows[lev-1], temp_gridDim, temp_blockDim);
    //     setToZero<<<temp_gridDim, temp_blockDim>>>( m_d_temp_matrix, m_num_rows[lev] * m_num_rows[lev-1]);
    //     RAP( d_value, d_index, m_max_row_size, d_r_value, d_r_index, m_r_max_row_size, d_p_value, d_p_index, m_p_max_row_size, m_d_temp_matrix, m_num_rows, lev);
    // }

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

