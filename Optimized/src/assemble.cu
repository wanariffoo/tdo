#include <iostream>
#include <cmath>
#include "../include/assemble.h"
#include "../include/cudakernels.h"


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

vector<size_t> Assembler::getNodeIndex()
{
    return m_node_index_;
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
    vector<size_t*> &d_node_index,
    size_t* &d_node_index_,
    ofstream& ofssbm)
{
    // benchmark output
    // ofstream ofssbm(filename, ios::out);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds;




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

            else if (m_N[0][0] == 6 && m_N[0][1] == 2 && m_N[0][2] == 1 )
            {
                max_row_size[0] = 18*3;
                p_max_row_size[0] = 8;
                r_max_row_size[0] = 18;
            }

            else if (m_N[0][0] == 6 && m_N[0][1] == 1 && m_N[0][2] == 2 )
            {
                max_row_size[0] = 18*3;
                p_max_row_size[0] = 8;
                r_max_row_size[0] = 18;
            }


            else
            {   
                throw(runtime_error("3D assembly not done yet for this grid size"));
            }
            
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

    // calculating the number of nodes per level
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

    // output file
    ofssbm << "Top-level grid size = { " << m_N[m_topLev][0];
    for ( int i = 1 ; i < m_dim ; ++i )
        ofssbm << ", " << m_N[m_topLev][i];
    
    ofssbm << " }" << endl;
    ofssbm << "Top-level number of rows = " << m_num_rows[m_topLev] << endl;
    ofssbm << "Number of Elements = " << m_numNodes[m_topLev] << endl;
    ofssbm << endl;
    ofssbm << "(All measurements in ms)" << endl;
    ofssbm << endl;
    ofssbm << "ASSEMBLER" << endl;

    
    // // allocating and copying the boundary condition indices on each level
    // m_d_bc_index.resize(m_numLevels);
    
    // for ( int i = 0 ; i < m_numLevels ; i++ )
    // {
    //     CUDA_CALL( cudaMalloc( (void**)&m_d_bc_index[i], sizeof(size_t) * m_bc_index[i].size()) );
    //     CUDA_CALL( cudaMemcpy( m_d_bc_index[i], &m_bc_index[i][0], sizeof(size_t) * m_bc_index[i].size(), cudaMemcpyHostToDevice) );
    // }



    // assembling the local stiffness matrix   
        cudaEventRecord(start);
    assembleLocal();
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        ofssbm << "assembleLocal() \t\t" << milliseconds << endl;
        


        // temporary ellpack arrays (non-transposed) for prol and rest
        vector<double*> d_p_value_;
        vector<size_t*> d_p_index_;
        vector<double*> d_r_value_;
        vector<size_t*> d_r_index_;
       

    //// assembling prolongation & restriction matrices
        cudaEventRecord(start);
    assembleProlMatrix_GPU(d_p_value_, d_p_index_, m_topLev);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        ofssbm << "assembleProlMatrix_GPU() \t" << milliseconds << endl;

        cudaEventRecord(start);
    assembleRestMatrix_GPU(d_r_value_, d_r_index_, d_p_value_, d_p_index_);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        ofssbm << "assembleRestMatrix_GPU() \t" << milliseconds << endl;

    // resizing the prolongation matrices according to the number of grid-levels for cuda
    d_p_value.resize( m_numLevels - 1 );
    d_p_index.resize( m_numLevels - 1 );

    // resizing the restriction matrices according to the number of grid-levels (transposed ELLPACK)
    d_r_value.resize( m_numLevels - 1 );
    d_r_index.resize( m_numLevels - 1 );

    // allocating and copying the value & index vectors to device
    for ( int lev = 0 ; lev < m_numLevels - 1 ; lev++ )
    {
        CUDA_CALL( cudaMalloc((void**)&d_p_value[lev], sizeof(double) * m_p_max_row_size[lev] * m_num_rows[lev+1] ) );
        CUDA_CALL( cudaMalloc((void**)&d_p_index[lev], sizeof(size_t) * m_p_max_row_size[lev] * m_num_rows[lev+1] ) );
        CUDA_CALL( cudaMalloc((void**)&d_r_value[lev], sizeof(double) * m_num_rows[lev] * m_r_max_row_size[lev] ) );
        CUDA_CALL( cudaMalloc((void**)&d_r_index[lev], sizeof(size_t) * m_num_rows[lev] * m_r_max_row_size[lev] ) );
    }

    // temp: matrices = (matrices_)^T
    for ( int lev = 0 ; lev < m_numLevels - 1 ; lev++ )
    {
        dim3 gridDim_; dim3 blockDim_;
        calculateDimensions( m_num_rows[lev+1]*m_p_max_row_size[lev], gridDim_, blockDim_);
        transposeELL<<<gridDim_,blockDim_>>>( d_p_index[lev], d_p_index_[lev], m_num_rows[lev+1], m_p_max_row_size[lev] );
        transposeELL<<<gridDim_,blockDim_>>>( d_p_value[lev], d_p_value_[lev], m_num_rows[lev+1], m_p_max_row_size[lev] );
        calculateDimensions( m_num_rows[lev]*m_r_max_row_size[lev], gridDim_, blockDim_);
        transposeELL<<<gridDim_,blockDim_>>>( d_r_index[lev], d_r_index_[lev], m_num_rows[lev], m_r_max_row_size[lev] );
        transposeELL<<<gridDim_,blockDim_>>>( d_r_value[lev], d_r_value_[lev], m_num_rows[lev], m_r_max_row_size[lev] );
       
    }

    // printLinearVector( d_r_index[0], m_r_max_row_size[0], m_num_rows[0] );
    // printLinearVector( d_r_index_[0], m_num_rows[0], m_r_max_row_size[0]);
    // printLinearVector( d_r_value[0], m_r_max_row_size[0], m_num_rows[0] );
    // printLinearVector( d_r_value_[0], m_num_rows[0], m_r_max_row_size[0]);
    // printELLrow_(0, d_r_value[0], d_r_index[0], r_max_row_size[0], num_rows[0], num_rows[1]);
    // printELLrow_(0, d_p_value[0], d_p_index[0], p_max_row_size[0], num_rows[1], num_rows[0]);
    // printELLrow(0, d_r_value_[0], d_r_index_[0], r_max_row_size[0], num_rows[0], num_rows[1]);
    // printELLrow(0, d_r_value_[0], d_r_index_[0], m_r_max_row_size[0], m_num_rows[0], m_num_rows[1]);
    // printELLrow(1, d_r_value[1], d_r_index[1], r_max_row_size[1], num_rows[1], num_rows[2]);

    //// adding nodes and elements to the top-level global grid
    for ( int i = 0 ; i < m_numNodes[m_topLev] ; ++i )
        m_node.push_back(Node(i));

    for ( int i = 0 ; i < m_numElements[m_topLev] ; ++i )
        m_element.push_back(Element(i));

    size_t numNodesIn2D = (m_N[m_topLev][0]+1)*(m_N[m_topLev][1]+1);


    // assigning the nodes to each element
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
        for ( int i = 0 ; i < m_numElements[m_topLev] ; i++ )
        {
            size_t elemcount_2D = (m_N[m_topLev][0])*(m_N[m_topLev][1]); 
            size_t gridsize_2D = (m_N[m_topLev][0]+1)*(m_N[m_topLev][1]+1);
            size_t multiplier = i / elemcount_2D;
            size_t base_id = i % elemcount_2D;

            m_element[i].addNode(&m_node[ base_id + base_id/m_N[m_topLev][0] + multiplier*gridsize_2D ]);   // lower left node
            m_element[i].addNode(&m_node[ base_id + base_id/m_N[m_topLev][0] + multiplier*gridsize_2D + 1]);   // lower right node
            m_element[i].addNode(&m_node[ base_id + base_id/m_N[m_topLev][0] + multiplier*gridsize_2D + m_N[m_topLev][0] + 1]);   // upper left node
            m_element[i].addNode(&m_node[ base_id + base_id/m_N[m_topLev][0] + multiplier*gridsize_2D + m_N[m_topLev][0] + 2]);   // upper right node
            
            // next layer
            m_element[i].addNode(&m_node[ base_id + base_id/m_N[m_topLev][0] + multiplier*gridsize_2D + gridsize_2D]);   // lower left node
            m_element[i].addNode(&m_node[ base_id + base_id/m_N[m_topLev][0] + multiplier*gridsize_2D + 1 + gridsize_2D]);   // lower right node
            m_element[i].addNode(&m_node[ base_id + base_id/m_N[m_topLev][0] + multiplier*gridsize_2D + m_N[m_topLev][0] + 1 + gridsize_2D]);   // upper left node
            m_element[i].addNode(&m_node[ base_id + base_id/m_N[m_topLev][0] + multiplier*gridsize_2D + m_N[m_topLev][0] + 2 + gridsize_2D]);   // upper right node

        }
    }
 


    
    m_node_index.resize(m_numElements[m_topLev]);
    d_node_index.resize(m_numElements[m_topLev]);
    for ( int elem = 0 ; elem < m_numElements[m_topLev] ; elem++ )
    {
        for ( int index = 0 ; index < pow(2, m_dim) ; index++ )
        {
            m_node_index[elem].push_back( m_element[elem].getNodeIndex(index) );
            m_node_index_.push_back( m_element[elem].getNodeIndex(index) );
        }
    }

    // // 
    // CUDA_CALL( cudaMalloc((void**)&d_node_index_, sizeof(double) * m_numElements[m_topLev] * pow(2, m_dim) ) );
    // CUDA_CALL( cudaMemcpy(d_node_index_, &node_index_[0], sizeof(double) * m_numElements[m_topLev] * pow(2, m_dim), cudaMemcpyHostToDevice) );
    

    



    // allocating and copying the design variable to device
    // design variable currently has initial values of rho
    CUDA_CALL( cudaMalloc((void**)&d_chi, sizeof(double) * m_numElements[m_topLev] ) );
    CUDA_CALL( cudaMemcpy(d_chi, &m_chi[0], sizeof(double) * m_numElements[m_topLev], cudaMemcpyHostToDevice) );

    // allocating and copying the (linear vector) local stiffness matrix to device
    CUDA_CALL( cudaMalloc((void**)&d_A_local, sizeof(double) * m_num_rows_l*m_num_rows_l ) );
    CUDA_CALL( cudaMemcpy( d_A_local, &m_A_local[0], sizeof(double) * m_num_rows_l*m_num_rows_l, cudaMemcpyHostToDevice) );

    
    // calculating the number of nodes in a local element
    size_t numNodes_local = pow(2,m_dim);

    // copying the node index vector to device
    for ( int i = 0 ; i < m_numElements[m_topLev] ; i++ )
    {
        CUDA_CALL( cudaMalloc( (void**)&d_node_index[i], sizeof(size_t) * numNodes_local) );
        CUDA_CALL( cudaMemcpy( d_node_index[i], &m_node_index[i][0], sizeof(size_t) * numNodes_local, cudaMemcpyHostToDevice) );
    }

    m_d_node_index = d_node_index;

    // allocating the global matrices of each level to device
    // matrices are empty for now, will be filled in later




    // resizing global matrices for each grid-level
    // these matrices are stored in a transposed ELLPACK format
    d_value.resize( m_numLevels );
    d_index.resize( m_numLevels );

    // temporary matrices for the global stiffness in non-transposed ELLPACK format
    vector<double*> d_value_( m_numLevels );
    vector<size_t*> d_index_( m_numLevels );

    for ( int lev = 0 ; lev < m_numLevels ; lev++ )
    {
        // transposed ellpack
        CUDA_CALL( cudaMalloc((void**)&d_value[lev], sizeof(double) * max_row_size[lev] * num_rows[lev] ) );
        CUDA_CALL( cudaMemset( d_value[lev], 0, sizeof(double) * num_rows[lev]*max_row_size[lev] ) );
        CUDA_CALL( cudaMalloc((void**)&d_index[lev], sizeof(size_t) * max_row_size[lev] * num_rows[lev] ) );
        CUDA_CALL( cudaMemset( d_index[lev], 0, sizeof(size_t) * num_rows[lev]*max_row_size[lev] ) );

        // non-transposed ellpack
        CUDA_CALL( cudaMalloc((void**)&d_value_[lev], sizeof(double) * max_row_size[lev] * num_rows[lev] ) );
        CUDA_CALL( cudaMemset( d_value_[lev], 0, sizeof(double) * num_rows[lev]*max_row_size[lev] ) );
        CUDA_CALL( cudaMalloc((void**)&d_index_[lev], sizeof(size_t) * max_row_size[lev] * num_rows[lev] ) );
        CUDA_CALL( cudaMemset( d_index_[lev], 0, sizeof(size_t) * num_rows[lev]*max_row_size[lev] ) );
    }
    
    
    // filling in global stiffness matrix's ELLPACK index vector for all levels
    dim3 index_gridDim;
    dim3 index_blockDim;

        cudaEventRecord(start);
    if ( m_dim == 2)
    {
        for (int lev = m_topLev ; lev >= 0 ; --lev )
        {
            calculateDimensions( num_rows[lev], index_gridDim, index_blockDim);
            fillIndexVector2D_GPU<<<index_gridDim,index_blockDim>>>(d_index_[lev], m_N[lev][0], m_N[lev][1], max_row_size[lev], num_rows[lev]);
        }
    }

    else
    {   
        for (int lev = m_topLev ; lev >= 0 ; --lev )
        {
            calculateDimensions( num_rows[lev], index_gridDim, index_blockDim);
            fillIndexVector3D_GPU<<<index_gridDim,index_blockDim>>>(d_index_[lev], m_N[lev][0], m_N[lev][1], m_N[lev][2], max_row_size[lev], num_rows[lev]);
        }
    }
    
    // copying to the transposed ellpack matrices
    for (int lev = 0 ; lev < m_numLevels ; ++lev )
    {
        calculateDimensions( num_rows[lev], index_gridDim, index_blockDim);
        transposeELL<<<index_gridDim, index_blockDim>>>( d_index[lev], d_index_[lev], m_num_rows[lev], m_max_row_size[lev] );
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    ofssbm << "fillIndexVector() \t\t" << milliseconds << endl;

    
    // printLinearVector( d_index[1], m_max_row_size[1], m_num_rows[1]);
    // printLinearVector( d_index_[1], m_num_rows[1], m_max_row_size[1]);

    
    

    //// filling in the top level global stiffness matrix
    // CUDA block size for assembling the global stiffness matrix
    dim3 l_blockDim(m_num_rows_l,m_num_rows_l,1);


    // filling in the local stiffness matrix from each element
    // the density distribution is included here as well
            cudaEventRecord(start);
    for ( int i = 0 ; i < m_numElements[m_topLev] ; ++i )
    {
        // assembleGrid2D_GPU<<<1,l_blockDim>>>( m_N[m_topLev][0], m_dim, &d_chi[i], d_A_local, &d_value_[m_topLev][0], &d_index_[m_topLev][0], max_row_size[m_topLev], m_num_rows_l, d_node_index[i], m_p);
        assembleGrid2D_GPU_<<<1,l_blockDim>>>( m_N[m_topLev][0], m_dim, &d_chi[i], d_A_local, m_num_rows_l, &d_value[m_topLev][0], &d_index[m_topLev][0], max_row_size[m_topLev], m_num_rows[m_topLev], d_node_index[i], m_p);
    }
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            milliseconds = 0;
            cudaEventElapsedTime(&milliseconds, start, stop);
            ofssbm << "assembleGrid2D_GPU() \t\t" << milliseconds << endl;

    // // DEBUG: temporary
    // transposeELL<<<index_gridDim, index_blockDim>>>( d_value[m_topLev], d_value_[m_topLev], m_num_rows[m_topLev], m_max_row_size[m_topLev] );
    
    // printELLrow_(m_topLev, d_value[m_topLev], d_index[m_topLev], max_row_size[m_topLev], num_rows[m_topLev], num_rows[m_topLev]);
    // printELLrow(1, d_value_[1], d_index_[1], max_row_size[1], num_rows[1], num_rows[1]);

    // printLinearVector( d_value[1], m_max_row_size[1], m_num_rows[1]);
    
    
    







    // // rr = A^T * xx
    // // NOTE: temporary test
    // vector<double> xx( m_num_rows[m_topLev], 1.0);
    // double* d_xx;
    // double* d_rr;
    
    // CUDA_CALL( cudaMalloc((void**)&d_rr, sizeof(double) * m_num_rows[m_topLev] ) );
    // CUDA_CALL( cudaMalloc((void**)&d_xx, sizeof(double) * m_num_rows[m_topLev] ) );
    // CUDA_CALL( cudaMemcpy(d_xx, &xx[0], sizeof(double) * m_num_rows[m_topLev], cudaMemcpyHostToDevice) );
    
    // // Ax = r (x = 1 hardcoded)
    // calculateDimensions( num_rows[m_topLev], index_gridDim, index_blockDim);

    // /// r = A^T * x, ( .., x, r )

    //     cudaEventRecord(start);
    // // ApplyTransposed_GPU_ <<<index_gridDim,index_blockDim>>>  (m_num_rows[m_topLev], m_max_row_size[m_topLev], d_value[m_topLev], d_index[m_topLev], d_xx, d_rr);
    // ApplyTransposed_GPU <<<index_gridDim,index_blockDim>>>  (m_num_rows[m_topLev], m_max_row_size[m_topLev], d_value_[m_topLev], d_index_[m_topLev], d_xx, d_rr);
    // // Apply_GPU_ <<<index_gridDim,index_blockDim>>>  (m_num_rows[m_topLev], m_max_row_size[m_topLev], d_value[m_topLev], d_index[m_topLev], d_xx, d_rr);
    //     cudaEventRecord(stop);
    //     cudaEventSynchronize(stop);
    //     milliseconds = 0;
    //     cudaEventElapsedTime(&milliseconds, start, stop);
    //     cout << "Apply_GPU_() \t\t" << milliseconds << endl;

    //     printVector_GPU<<<1, num_rows[m_topLev]>>>( d_rr, num_rows[m_topLev] );


    //     cudaEventRecord(start);
    // Apply_GPU <<<index_gridDim,index_blockDim>>>  (m_num_rows[m_topLev], m_max_row_size[m_topLev], d_value_[m_topLev], d_index_[m_topLev], d_xx, d_rr);
    //     cudaEventRecord(stop);
    //     cudaEventSynchronize(stop);
    //     milliseconds = 0;
    //     cudaEventElapsedTime(&milliseconds, start, stop);
    //     cout << "Apply_GPU() \t\t" << milliseconds << endl;

    // printLinearVector( d_value[1], m_max_row_size[1], m_num_rows[1]);
    // printLinearVector( d_value_[1], m_num_rows[1], m_max_row_size[1] );

    














    // calculating the needed cuda 2D grid size for the global assembly
    dim3 g_gridDim;
    dim3 g_blockDim;           
        
    // apply boundary conditions to the global stiffness matrix
    calculateDimensions( m_num_rows[m_topLev], g_gridDim, g_blockDim);
        cudaEventRecord(start);
    for ( int i = 0 ; i < m_bc_index[m_topLev].size() ; i++ )
    {
        applyMatrixBC_GPU_<<<g_gridDim,g_blockDim>>>(&d_value[m_topLev][0], &d_index[m_topLev][0], m_max_row_size[m_topLev], m_bc_index[m_topLev][i], m_num_rows[m_topLev], m_num_rows[m_topLev] );
    }
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            milliseconds = 0;
            cudaEventElapsedTime(&milliseconds, start, stop);
            ofssbm << "applyMatrixBC_GPU() \t\t" << milliseconds << endl;

    //// obtaining the coarse stiffness matrices of each lower grid level
    dim3 temp_gridDim;
    dim3 temp_blockDim;
    
    // A_coarse = R * A_fine * P
            cudaEventRecord(start);
    for ( int lev = m_topLev ; lev != 0 ; lev--)
    {
        calculateDimensions(m_num_rows[lev], temp_gridDim, temp_blockDim);
        PTAP<<<temp_gridDim, temp_blockDim>>>(d_value[lev], d_index[lev], max_row_size[lev], num_rows[lev], 
                                    d_value[lev-1], d_index[lev-1], max_row_size[lev-1], num_rows[lev-1], 
                                    d_p_value[lev-1], d_p_index[lev-1], p_max_row_size[lev-1]);
    }

            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            milliseconds = 0;
            cudaEventElapsedTime(&milliseconds, start, stop);
            ofssbm << "PTAP() \t\t\t\t" << milliseconds << endl;


            
            
            //TODO: cudaFree the temp matrices
            


    // // delete temporary matrices

    // for( int i = 0 ; i < m_numLevels ; i++ )
    // {
    //     CUDA_CALL( cudaFree( d_value_[i]) );
    //     CUDA_CALL( cudaFree( d_index_[i]) );
    // }

    // for( int i = 0 ; i < m_numLevels - 1; i++ )
    // {
    //     CUDA_CALL( cudaFree( d_p_value_[i]) );
    //     CUDA_CALL( cudaFree( d_p_index_[i]) );
    //     CUDA_CALL( cudaFree( d_r_value_[i]) );
    //     CUDA_CALL( cudaFree( d_r_index_[i]) );
    // }
    
            
            


    // // NOTE: temporary test
    // vector<double> xx( m_num_rows[1], 1.0);
    // double* d_xx;
    // double* d_rr;
    
    // CUDA_CALL( cudaMalloc((void**)&d_xx, sizeof(double) * m_num_rows[1] ) );
    // CUDA_CALL( cudaMalloc((void**)&d_rr, sizeof(double) * m_num_rows[1] ) );
    // CUDA_CALL( cudaMemcpy(d_xx, &xx[1], sizeof(double) * m_num_rows[1], cudaMemcpyHostToDevice) );
    
    // // Ax = r
    // calculateDimensions( num_rows[1], index_gridDim, index_blockDim);
    // ApplyTransposed_GPU_ <<<index_gridDim,index_blockDim>>>  (m_num_rows[1], m_max_row_size[1], d_value[1], d_index[1], d_xx, d_rr);
    
    // printVector_GPU<<<1,num_rows[1]>>>( d_rr, num_rows[1] );
            
            
            
            
    // printLinearVector( d_index_[1], m_num_rows[1], m_max_row_size[1]);
    // printLinearVector( d_r_index[1], m_num_rows[1], m_r_max_row_size[1]);
    // printELLrow(0, d_value_[0], d_index_[0], max_row_size[0], num_rows[0], num_rows[0]);
    // printELLrow(1, d_value_[1], d_index_[1], max_row_size[1], num_rows[1], num_rows[1]);
    // printELLrow(m_topLev-1, d_value_[m_topLev-1], d_index_[m_topLev-1], max_row_size[m_topLev-1], num_rows[m_topLev-1], num_rows[m_topLev-1]);
    // printELLrow_(0, d_value[0], d_index[0], max_row_size[0], num_rows[0], num_rows[0]);
    // printELLrow_(1, d_value[1], d_index[1], max_row_size[1], num_rows[1], num_rows[1]);
    // printELLrow_(m_topLev-1, d_value[m_topLev-1], d_index[m_topLev-1], max_row_size[m_topLev-1], num_rows[m_topLev-1], num_rows[m_topLev-1]);
    // // printELLrow(1, d_value[1], d_index[1], max_row_size[1], num_rows[1], num_rows[1]);
    // // printELLrow(2, d_value[2], d_index[2], max_row_size[2], num_rows[2], num_rows[2]);


    return true;

}




bool Assembler::assembleLocal()
{

    double det_jacobi;
    double inv_jacobi;

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

        det_jacobi = pow(m_h/2, m_dim);
        inv_jacobi = 1 / (m_h/2);
        
        // loop through each set of gauss points
        for ( int i = 0 ; i < 4 ; ++i )
        {
            // resetting of vectors for each loop calculation
            A_.clear();
            A_.resize(3, vector<double>(8));
            
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
    
        // isotropic linear elastic tensor
        double lambda = (m_youngMod * m_poisson) / ((1+m_poisson)*(1-2*m_poisson));
        double mu = m_youngMod / ( 2 * (1+m_poisson) );

        vector<vector<double>> E (6, vector <double> (6, 0.0));
        vector<vector<double>> A_ (6, vector <double> (24, 0.0));

        E[0][0] = E[1][1] = E[2][2] = lambda + 2*mu;
        E[1][0] = E[2][0] = E[0][1] = E[0][2] = E[1][2] = E[2][1] = lambda;
        E[3][3] = E[4][4] = E[5][5] = mu;

        // jacobi
        vector<vector<double>> J (3, vector <double> (3, 0.0));
        J[0][0] = m_h / 2;
        J[1][1] = m_h / 2;
        J[2][2] = m_h / 2;
        
        det_jacobi = pow(m_h/2, m_dim);
        inv_jacobi = 1 / (m_h/2);

        // 8 gauss points
        vector<double> r_ = {-(1/sqrt(3)), (1/sqrt(3))};
        vector<double> s_ = {-(1/sqrt(3)), (1/sqrt(3))};
        vector<double> t_ = {-(1/sqrt(3)), (1/sqrt(3))};

        // B-matrix
        vector<vector<double>> B(6, vector <double> (24, 0));

        
        //// loop through each gauss points
        // nodes' natural coordinates
        vector<double> r = {-1, 1};
        vector<double> s = {-1, 1};
        vector<double> t = {-1, 1};

        

        for ( int _t = 0 ; _t < 2 ; _t++ )
        {
            for ( int _s = 0 ; _s < 2 ; _s++ )
            {
                for ( int _r = 0 ; _r < 2 ; _r++ )
                {
                    A_.clear();
                    A_.resize(6, vector<double>(24));   
                    int i;

                    // node 0
                    i=0;
                    B[0][i*3]   = (inv_jacobi/8) * r[0] * ( 1 + (s[0]*s_[_s]) ) * ( 1 + (t[0]*t_[_t]) );
                    B[1][i*3+1] = (inv_jacobi/8) * s[0] * ( 1 + (r[0]*r_[_r]) ) * ( 1 + (t[0]*t_[_t]) );
                    B[2][i*3+2] = (inv_jacobi/8) * t[0] * ( 1 + (r[0]*r_[_r]) ) * ( 1 + (s[0]*s_[_s]) );
                    B[3][i*3]   = B[1][i*3+1];
                    B[3][i*3+1] = B[0][i*3];
                    B[4][i*3+1] = B[2][i*3+2];
                    B[4][i*3+2] = B[1][i*3+1];
                    B[5][i*3]   = B[2][i*3+2];
                    B[5][i*3+2] = B[0][i*3];

                    // node 1
                    i=1;
                    B[0][i*3]   = (inv_jacobi/8) * r[1] * ( 1 + (s[0]*s_[_s]) ) * ( 1 + (t[0]*t_[_t]) );
                    B[1][i*3+1] = (inv_jacobi/8) * s[0] * ( 1 + (r[1]*r_[_r]) ) * ( 1 + (t[0]*t_[_t]) );
                    B[2][i*3+2] = (inv_jacobi/8) * t[0] * ( 1 + (r[1]*r_[_r]) ) * ( 1 + (s[0]*s_[_s]) );
                    B[3][i*3]   = B[1][i*3+1];
                    B[3][i*3+1] = B[0][i*3];
                    B[4][i*3+1] = B[2][i*3+2];
                    B[4][i*3+2] = B[1][i*3+1];
                    B[5][i*3]   = B[2][i*3+2];
                    B[5][i*3+2] = B[0][i*3];

                    // node 2
                    i=2;
                    B[0][i*3]   = (inv_jacobi/8) * r[0] * ( 1 + (s[1]*s_[_s]) ) * ( 1 + (t[0]*t_[_t]) );
                    B[1][i*3+1] = (inv_jacobi/8) * s[1] * ( 1 + (r[0]*r_[_r]) ) * ( 1 + (t[0]*t_[_t]) );
                    B[2][i*3+2] = (inv_jacobi/8) * t[0] * ( 1 + (r[0]*r_[_r]) ) * ( 1 + (s[1]*s_[_s]) );
                    B[3][i*3]   = B[1][i*3+1];
                    B[3][i*3+1] = B[0][i*3];
                    B[4][i*3+1] = B[2][i*3+2];
                    B[4][i*3+2] = B[1][i*3+1];
                    B[5][i*3]   = B[2][i*3+2];
                    B[5][i*3+2] = B[0][i*3];

                    // node 3
                    i=3;
                    B[0][i*3]   = (inv_jacobi/8) * r[1] * ( 1 + (s[1]*s_[_s]) ) * ( 1 + (t[0]*t_[_t]) );
                    B[1][i*3+1] = (inv_jacobi/8) * s[1] * ( 1 + (r[1]*r_[_r]) ) * ( 1 + (t[0]*t_[_t]) );
                    B[2][i*3+2] = (inv_jacobi/8) * t[0] * ( 1 + (r[1]*r_[_r]) ) * ( 1 + (s[1]*s_[_s]) );
                    B[3][i*3]   = B[1][i*3+1];
                    B[3][i*3+1] = B[0][i*3];
                    B[4][i*3+1] = B[2][i*3+2];
                    B[4][i*3+2] = B[1][i*3+1];
                    B[5][i*3]   = B[2][i*3+2];
                    B[5][i*3+2] = B[0][i*3];

                    // node 4
                    i=4;
                    B[0][i*3]   = (inv_jacobi/8) * r[0] * ( 1 + (s[0]*s_[_s]) ) * ( 1 + (t[1]*t_[_t]) );
                    B[1][i*3+1] = (inv_jacobi/8) * s[0] * ( 1 + (r[0]*r_[_r]) ) * ( 1 + (t[1]*t_[_t]) );
                    B[2][i*3+2] = (inv_jacobi/8) * t[1] * ( 1 + (r[0]*r_[_r]) ) * ( 1 + (s[0]*s_[_s]) );
                    B[3][i*3]   = B[1][i*3+1];
                    B[3][i*3+1] = B[0][i*3];
                    B[4][i*3+1] = B[2][i*3+2];
                    B[4][i*3+2] = B[1][i*3+1];
                    B[5][i*3]   = B[2][i*3+2];
                    B[5][i*3+2] = B[0][i*3];

                    // node 5
                    i=5;
                    B[0][i*3]   = (inv_jacobi/8) * r[1] * ( 1 + (s[0]*s_[_s]) ) * ( 1 + (t[1]*t_[_t]) );
                    B[1][i*3+1] = (inv_jacobi/8) * s[0] * ( 1 + (r[1]*r_[_r]) ) * ( 1 + (t[1]*t_[_t]) );
                    B[2][i*3+2] = (inv_jacobi/8) * t[1] * ( 1 + (r[1]*r_[_r]) ) * ( 1 + (s[0]*s_[_s]) );
                    B[3][i*3]   = B[1][i*3+1];
                    B[3][i*3+1] = B[0][i*3];
                    B[4][i*3+1] = B[2][i*3+2];
                    B[4][i*3+2] = B[1][i*3+1];
                    B[5][i*3]   = B[2][i*3+2];
                    B[5][i*3+2] = B[0][i*3];

                    // node 6
                    i=6;
                    B[0][i*3]   = (inv_jacobi/8) * r[0] * ( 1 + (s[1]*s_[_s]) ) * ( 1 + (t[1]*t_[_t]) );
                    B[1][i*3+1] = (inv_jacobi/8) * s[1] * ( 1 + (r[0]*r_[_r]) ) * ( 1 + (t[1]*t_[_t]) );
                    B[2][i*3+2] = (inv_jacobi/8) * t[1] * ( 1 + (r[0]*r_[_r]) ) * ( 1 + (s[1]*s_[_s]) );
                    B[3][i*3]   = B[1][i*3+1];
                    B[3][i*3+1] = B[0][i*3];
                    B[4][i*3+1] = B[2][i*3+2];
                    B[4][i*3+2] = B[1][i*3+1];
                    B[5][i*3]   = B[2][i*3+2];
                    B[5][i*3+2] = B[0][i*3];

                    // node 7
                    i=7;
                    B[0][i*3]   = (inv_jacobi/8) * r[1] * ( 1 + (s[1]*s_[_s]) ) * ( 1 + (t[1]*t_[_t]) );
                    B[1][i*3+1] = (inv_jacobi/8) * s[1] * ( 1 + (r[1]*r_[_r]) ) * ( 1 + (t[1]*t_[_t]) );
                    B[2][i*3+2] = (inv_jacobi/8) * t[1] * ( 1 + (r[1]*r_[_r]) ) * ( 1 + (s[1]*s_[_s]) );
                    B[3][i*3]   = B[1][i*3+1];
                    B[3][i*3+1] = B[0][i*3];
                    B[4][i*3+1] = B[2][i*3+2];
                    B[4][i*3+2] = B[1][i*3+1];
                    B[5][i*3]   = B[2][i*3+2];
                    B[5][i*3+2] = B[0][i*3];


                    //// A_local = B^T * E * B * det(J)

                    // A_ = E * B
                    for ( int i = 0 ; i < 6 ; i++ )
                    {
                        for( int j = 0 ; j < 24 ; j++ )
                        {
                            for ( int k = 0 ; k < 6 ; k++)
                                A_[i][j] += E[i][k] * B[k][j];
                        }
                    }
                    
                    // A_local = B^T * A_ * det(J)
                    for ( int i = 0 ; i < 24 ; i++ )
                    {
                        for( int j = 0 ; j < 24 ; j++ )
                        {
                            for ( int k = 0 ; k < 6 ; k++){
                                
                                m_A_local[j + i*m_num_rows_l] += B[k][i] * A_[k][j] * det_jacobi;  
                            }
                        }
                    }
                }
            }
        }


        
        

        }

        return true;
}

double Assembler::valueAt(size_t row, size_t col)
{
    return m_A_local[col + row*m_num_rows_l];
}

// assembles prolongation matrices into a transposed ellpack format
bool Assembler::assembleProlMatrix_GPU(
    vector<double*> &d_p_value_, 
    vector<size_t*> &d_p_index_, 
    size_t lev)
{
    
    d_p_value_.resize( m_numLevels - 1 );
    d_p_index_.resize( m_numLevels - 1 );

    // allocating and copying the value & index vectors to device
    for ( int lev = 0 ; lev < m_numLevels - 1 ; lev++ )
    {
        CUDA_CALL( cudaMalloc((void**)&d_p_value_[lev], sizeof(double) * m_p_max_row_size[lev] * m_num_rows[lev+1] ) );
        CUDA_CALL( cudaMalloc((void**)&d_p_index_[lev], sizeof(size_t) * m_p_max_row_size[lev] * m_num_rows[lev+1] ) );
    }




    dim3 gridDim;
    dim3 blockDim;

    if ( m_dim == 2)
    {
        // fill in prolongation matrix's ELLPACK index array (non-transposed)
        for ( int lev = 0 ; lev < m_numLevels - 1 ; lev++ )
        {
            calculateDimensions(m_num_rows[lev+1], gridDim, blockDim);
            fillProlMatrix2D_GPU<<<gridDim,blockDim>>>( d_p_value_[lev], d_p_index_[lev], m_N[lev+1][0], m_N[lev+1][1], m_p_max_row_size[lev], m_num_rows[lev+1], m_num_rows[lev]);
        }
    }

    // m_dim = 3
    else
    {
        for ( int lev = 0 ; lev < m_numLevels - 1 ; lev++ )
        {
            calculateDimensions(m_num_rows[lev+1], gridDim, blockDim);
            fillProlMatrix3D_GPU<<<gridDim,blockDim>>>( d_p_value_[lev], d_p_index_[lev], m_N[lev+1][0], m_N[lev+1][1], m_N[lev+1][2], m_p_max_row_size[lev], m_num_rows[lev+1], m_num_rows[lev]);
        }
    }
    
    // // prolmat = prolmat_^T
    // for ( int lev = 0 ; lev < m_numLevels - 1 ; lev++ )
    //     transposeELL<<<gridDim,blockDim>>>( d_p_index[lev], d_p_index_[lev], m_num_rows[lev], m_p_max_row_size[lev] );
        
    // printLinearVector( d_p_index_[0], m_num_rows[1], m_p_max_row_size[0]);
    // printLinearVector( d_p_index[0], m_p_max_row_size[0], m_num_rows[0]);
    
    // // applying boundary conditions on the prolongation matrices on each level
    // for ( int lev = 1 ; lev < m_numLevels ; lev++ )
    // {
    //     calculateDimensions(m_bc_index[lev-1].size(), gridDim, blockDim);
    //     applyProlMatrixBC_GPU<<<gridDim,blockDim>>>(&d_p_value[lev-1][0], &d_p_index[lev-1][0], m_p_max_row_size[lev-1], m_d_bc_index[lev-1], m_num_rows[lev], m_num_rows[lev-1], m_bc_index[lev-1].size() );
    // }

    // // TODO:
    // // deallocating the temporary arrays
    // for ( int lev = 0 ; lev < m_numLevels - 1 ; lev++ )
    // {
    //     CUDA_CALL( cudaFree(d_p_value_[lev]) );
    //     CUDA_CALL( cudaFree(d_p_index_[lev]) );
    // }


    return true;

}

bool Assembler::assembleRestMatrix_GPU(
    vector<double*> &d_r_value_, 
    vector<size_t*> &d_r_index_, 
    vector<double*> &d_p_value_, 
    vector<size_t*> &d_p_index_)
{

    d_r_value_.resize( m_numLevels - 1 );
    d_r_index_.resize( m_numLevels - 1 );

    // memcpy to device
    for ( int lev = 0 ; lev < m_numLevels - 1 ; lev++ )
    {
        // non-transposed
        CUDA_CALL( cudaMalloc((void**)&d_r_value_[lev], sizeof(double) * m_num_rows[lev] * m_r_max_row_size[lev] ) );
        CUDA_CALL( cudaMemset( d_r_value_[lev], 0, sizeof(double) * m_num_rows[lev] * m_r_max_row_size[lev] ) );
        CUDA_CALL( cudaMalloc((void**)&d_r_index_[lev], sizeof(size_t) * m_num_rows[lev] * m_r_max_row_size[lev] ) );
        CUDA_CALL( cudaMemset( d_r_index_[lev], 0, sizeof(size_t) * m_num_rows[lev] * m_r_max_row_size[lev] ) );
    }

    dim3 gridDim;
    dim3 blockDim;

    if ( m_dim == 2)
    {
        // fill in restriction matrix's ELLPACK index vector
        for ( int lev = 0 ; lev < m_numLevels - 1 ; lev++ )
        {
            calculateDimensions(m_numNodes[lev]*m_dim, gridDim, blockDim);
            fillIndexVectorRest2D_GPU<<<gridDim,blockDim>>>(d_r_index_[lev], m_N[lev][0], m_N[lev][1], m_r_max_row_size[lev], m_num_rows[lev], m_num_rows[lev+1]);
        }

        // fill in restriction matrix's values, taken from prolongation matrix
        for ( int lev = 0 ; lev < m_numLevels - 1 ; lev++ )
        {
            calculateDimensions2D(m_num_rows[lev], m_num_rows[lev+1], gridDim, blockDim);
            fillRestMatrix<<<gridDim, blockDim>>>(d_r_value_[lev], d_r_index_[lev], m_r_max_row_size[lev], d_p_value_[lev], d_p_index_[lev], m_p_max_row_size[lev], m_num_rows[lev], m_num_rows[lev+1]);
        }
    }

    else
    {
        // fill in restriction matrix's ELLPACK index vector
        for ( int lev = 0 ; lev < m_numLevels - 1 ; lev++ )
        {
            calculateDimensions(m_numNodes[lev]*m_dim, gridDim, blockDim);
            fillIndexVectorRest3D_GPU<<<gridDim,blockDim>>>(d_r_index_[lev], m_N[lev][0], m_N[lev][1], m_N[lev][2], m_r_max_row_size[lev], m_num_rows[lev], m_num_rows[lev+1]);
        }

        // fill in restriction matrix's values, taken from prolongation matrix
        for ( int lev = 0 ; lev < m_numLevels - 1 ; lev++ )
        {
            calculateDimensions2D(m_num_rows[lev], m_num_rows[lev+1], gridDim, blockDim);
            fillRestMatrix<<<gridDim, blockDim>>>(d_r_value_[lev], d_r_index_[lev], m_r_max_row_size[lev], d_p_value_[lev], d_p_index_[lev], m_p_max_row_size[lev], m_num_rows[lev], m_num_rows[lev+1]);
        }
    }
    

    // // restmat = restmat_^T
    // for ( int lev = 0 ; lev < m_numLevels - 1 ; lev++ ){

    //     transposeELL<<<gridDim,blockDim>>>( d_r_index[lev], d_r_index_[lev], m_num_rows[lev], m_r_max_row_size[lev] );
    //     transposeELL<<<gridDim,blockDim>>>( d_r_value[lev], d_r_value_[lev], m_num_rows[lev], m_r_max_row_size[lev] );
    // }
        
    // // printLinearVector( d_p_index_[0], m_num_rows[0], m_p_max_row_size[0]);
    // printLinearVector( d_r_index_[0], m_num_rows[0], m_r_max_row_size[0] );
    
    // printELLrow(0, d_r_value_[0], d_r_index_[0], m_r_max_row_size[0], m_num_rows[0], m_num_rows[1]);


   
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
        
    // assemble the global stiffness matrix on the finest grid with the updated chi of each element
    dim3 l_blockDim(m_num_rows_l,m_num_rows_l,1);
    for ( int i = 0 ; i < m_numElements[m_topLev] ; ++i )
        assembleGrid2D_GPU_<<<1,l_blockDim>>>( m_N[m_topLev][0], m_dim, &d_chi[i], d_A_local, m_num_rows_l, &d_value[m_topLev][0], &d_index[m_topLev][0], m_max_row_size[m_topLev], m_num_rows[m_topLev], m_d_node_index[i], m_p);
    
    // apply boundary conditions to the global stiffness matrix
    dim3 g_gridDim;
    dim3 g_blockDim;
    calculateDimensions( m_num_rows[m_topLev], g_gridDim, g_blockDim);
    for ( int i = 0 ; i < m_bc_index[m_topLev].size() ; i++ )
        applyMatrixBC_GPU_<<<g_gridDim,g_blockDim>>>(&d_value[m_topLev][0], &d_index[m_topLev][0], m_max_row_size[m_topLev], m_bc_index[m_topLev][i], m_num_rows[m_topLev], m_num_rows[m_topLev] );


    //// obtaining the coarse stiffness matrices of each lower grid level
    dim3 temp_gridDim;
    dim3 temp_blockDim;
    
    // A_coarse = R * A_fine * P
        for ( int lev = m_topLev ; lev != 0 ; lev--)
        {
            calculateDimensions(m_num_rows[lev], temp_gridDim, temp_blockDim);
            PTAP<<<temp_gridDim, temp_blockDim>>>(d_value[lev], d_index[lev], m_max_row_size[lev], m_num_rows[lev], 
                                        d_value[lev-1], d_index[lev-1], m_max_row_size[lev-1], m_num_rows[lev-1], 
                                        d_p_value[lev-1], d_p_index[lev-1], m_p_max_row_size[lev-1]);
        }


 


    return true;


    // NOTE: old
    // // cout << "Reassembling global stiffness matrix on finest grid" << endl;
    //     // assemble the global stiffness matrix on the finest grid with the updated chi of each element
    //     for ( int i = 0 ; i < m_numElements[m_topLev] ; ++i )
    //         assembleGrid2D_GPU<<<1,l_blockDim>>>( m_N[m_topLev][0], m_dim, &d_chi[i], d_A_local, &d_value[m_topLev][0], &d_index[m_topLev][0], m_max_row_size[m_topLev], m_num_rows_l, m_d_node_index[i], m_p);
    // // cuda 2D grid size for the boundary condition on the global stiffness matrix
    // dim3 g_gridDim;
    // dim3 g_blockDim;

    // applying the boundary conditions on the global stiffness matrix
    
    // calculateDimensions( m_bc_index[m_topLev].size(), g_gridDim, g_blockDim);
    // applyMatrixBC_GPU_<<<g_gridDim,g_blockDim>>>(&d_value[m_topLev][0], &d_index[m_topLev][0], m_max_row_size[m_topLev], m_d_bc_index[m_topLev], m_num_rows[m_topLev], m_bc_index[m_topLev].size() );

    
    // calculateDimensions2D( m_num_rows[m_topLev], m_num_rows[m_topLev], g_gridDim, g_blockDim);
    // for ( int i = 0 ; i < m_bc_index[m_topLev].size() ; i++ )
    //     applyMatrixBC_GPU<<<g_gridDim,g_blockDim,0,m_streams[i]>>>(&d_value[m_topLev][0], &d_index[m_topLev][0], m_max_row_size[m_topLev], m_bc_index[m_topLev][i], m_num_rows[m_topLev], m_num_rows[m_topLev] );

    // calculateDimensions2D( m_num_rows[m_topLev], m_num_rows[m_topLev], g_gridDim, g_blockDim);
    // for ( int i = 0 ; i < m_bc_index[m_topLev].size() ; i++ )
    //     applyMatrixBC_GPU_test<<<g_gridDim,g_blockDim>>>(&d_value[m_topLev][0], &d_index[m_topLev][0], m_max_row_size[m_topLev], m_bc_index[m_topLev][i], m_num_rows[m_topLev], m_num_rows[m_topLev] );
            
    //    // cuda 2D grid size to obtain the coarse matrices
    //    dim3 temp_gridDim;
    //    dim3 temp_blockDim;

    //    // A_coarse = R * A_fine * P
    //    for ( int lev = m_topLev ; lev != 0 ; lev--)
    //    {
    //        // cout << "Assembling coarse stiffness matrices" << endl;  

    //        calculateDimensions2D( m_num_rows[lev-1], m_num_rows[lev-1], temp_gridDim, temp_blockDim);
    //        RAP_<<<temp_gridDim,temp_blockDim>>>(   d_value[lev], d_index[lev], m_max_row_size[lev], m_num_rows[lev], 
    //                                                d_value[lev-1], d_index[lev-1], m_max_row_size[lev-1], m_num_rows[lev-1], 
    //                                                d_r_value[lev-1], d_r_index[lev-1], m_r_max_row_size[lev-1],
    //                                                d_p_value[lev-1], d_p_index[lev-1], m_p_max_row_size[lev-1], lev-1);
    //        cudaDeviceSynchronize();
    //    }

}

