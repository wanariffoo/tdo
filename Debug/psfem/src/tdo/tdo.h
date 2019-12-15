#ifndef TDO_H
#define TDO_H





class Node
{
public:
    Node (int id) : m_index(id){}


    void setXCoor(float x) { m_coo[0] = x;}
    void setYCoor(float y) { m_coo[1] = y;}
    float getXCoor(float x) { return m_coo[0];}
    float getYCoor(float y) { return m_coo[1];}

    void printCoor()
    {
        std::cout << "node [" << m_index << "] = ( " << m_coo[0] << ", " << m_coo[1] << " )" << std::endl;
    }
    int index() 
    { 
        return m_index; 
    }

   

private:
    int m_index;
    float m_coo[2];
    int m_dof[2];
    // vector<int> m_dof(2);
};



class Element
{
public:
    
    // global 
    Element()
    {
        m_vValue.resize(72);
        m_vIndex.resize(72);

        m_max_row_size = 4;
        m_num_rows = 18;
    }
    
    // local element
    Element(std::size_t ind) : m_index(ind)  // TODO: change int -> size_t
    {   

            m_vValue.resize(25);
            m_vIndex.resize(25);
            
            m_vValue = {4, 	1, 	0, 	1, 	4, 	1, 	1, 	4, 	1, 	1, 	4, 	1, 	1, 	4, 	1, 	1, 	4, 	1, 	1, 	4, 	1, 	1, 	4, 	0};
            m_vIndex = {0, 	1, 	8, 	0, 	1, 	2, 	1, 	2, 	3, 	2, 	3, 	4, 	3, 	4, 	5, 	4, 	5, 	6, 	5, 	6, 	7, 	6, 	7, 	8};

            m_max_row_size = 3;
            m_num_rows = 4;

    }
    

    std::size_t index()
    {
        return m_index;
    }

    void addNode(Node *x)
    {
        m_node.push_back(x);
        m_node_index_list.push_back(x->index());
    }

    void printNodes()
    {
        std::cout << "Element " << m_index << std::endl;
        for ( int i = 0 ; i < m_node.size() ; ++i )
            m_node[i]->printCoor();
    }

    double* getValueAddress() { return &m_vValue[0]; }
    std::size_t* getIndexAddress() { return &m_vIndex[0]; }

    std::size_t* getNodeGlobalIndex() { return &m_node_index_list[0]; }

    std::size_t max_row_size() { return m_max_row_size; }
    std::size_t num_rows() { return m_num_rows; }

    int nodeIndex(int i)
    {
        return m_node[i]->index();
    }

    double operator()(std::size_t x, std::size_t j) 
    {
        return m_K[x][j];
    }



private:
    std::vector<Node*> m_node;
    size_t m_index;
    size_t m_max_row_size;
    size_t m_num_rows;
    std::vector<size_t> m_node_index_list;
    double m_rho;

    double m_K[8][8];   // TODO: change 8 to dimension-friendly variable
    std::vector<double> m_vValue;
    std::vector<size_t> m_vIndex;


};

__global__ 
void getMaxRowSize(double *array, size_t *max, int *mutex, size_t n)
{
	unsigned int id = threadIdx.x + blockIdx.x*blockDim.x; // 0 - 24
	unsigned int stride = gridDim.x*blockDim.x; // 5
	// unsigned int offset = 0;

    __shared__ size_t local_nnz[1024];

    local_nnz[id] = 0;

    // CHECK: something's fishy here
    // get nnz of each row
        for ( int j = 0 ; j < stride ; j++)
        {
            if ( array[j + stride*id] != 0)
            local_nnz[id]++;
        }

	// reduction
	unsigned int i = blockDim.x/2;
    while(i != 0)
    {
		if(threadIdx.x < i){
			local_nnz[threadIdx.x] = fmaxf(local_nnz[threadIdx.x], local_nnz[threadIdx.x + i]);
		}
		__syncthreads();
		i /= 2;
	}

    if(threadIdx.x == 0)
    {
		while(atomicCAS(mutex,0,1) != 0);  //lock
		*max = fmaxf(*max, local_nnz[0]);
        atomicExch(mutex, 0);  //unlock
    }

}


// __global__
// void convertToELL(double *coo, double *ell_val, std::size_t *ell_ind, std::size_t max_row_size, std::size_t N)
// {
// 	unsigned int id = threadIdx.x + blockIdx.x*blockDim.x; // 0 - 5
//     unsigned int stride = gridDim.x*blockDim.x; // 6
    
//     extern __shared__ std::size_t temp[];
    
        
//         // Inputting the values into ELLPACK index vector

        
//         // fill in temp vector with values of COO's each row

//         for ( int i = 0 ; i < N ; ++i )
//         {
//             if(id == 0)
//                 printf("id*stride + i = %d\n", id*stride + i);

//             if ( coo[ id*stride + i ] == 0 )    // TODO: replace 0.0 with something else
//                 temp[ id*stride + i ] = N;

//             else
//                 temp[ id*stride + i ] = i;
//         }
       
//         int ind_counter = 0;
//         for ( int i = 0 ; i < N || ind_counter == max_row_size ; i++ )
//         {
//             if ( temp[ id*stride + i ] != N )
//             {
//                 temp[ id*stride + ind_counter ] = temp[ id*stride + i ];
//                 // temp[ id*stride + i ] = N;
//                 ind_counter++;
//             }
            
//             else{}
//         }
        

             
//     std::size_t counter = 0;

//     // NOTE: potentially, include arranging the temp[] or ell_ind() here as well?
//     // Compressing the COO, so that the NNZ are located in the first columns
//     for ( int i = 0 ; i < stride ; i++ )
//     {
//         if ( coo[ i + stride*id ] != 0 )    // TODO: change from 0.0 to abs(value) > 0
//         {
//             coo[counter + stride*id] = coo[ i + stride*id ];

//             if ( counter + stride*id != i + stride*id )
//                 coo[ i + stride*id ] = 0;

//             // coo[ i + stride*id ] = 0;
//             counter++;
//         }
        
//         else{}
//     }

//     // Inputting the values into ELLPACK value vector
//     if ( id < max_row_size )
//     {
//         for ( int i = 0 ; i < N ; ++i )
//         {
//             ell_val[ id*stride + i ] = coo [ id + stride*i ];
//             ell_ind[ id*stride + i ] = temp [ id + stride*i ];
//         }
//     }

// }

// returns value at A(x,y)
__device__
double valueAt(std::size_t x, std::size_t y, double* vValue, std::size_t* vIndex, std::size_t max_row_size)
{
    for(size_t k = 0; k < max_row_size; ++k)
    {
        if(vIndex[x * max_row_size + k] == y)
            return vValue[x * max_row_size + k];
    }

    return 0.0;
}

// A(x,y) = value
__device__
void setAt( std::size_t x, std::size_t y, double* vValue, std::size_t* vIndex, std::size_t max_row_size, double value )
{
    for(std::size_t k = 0; k < max_row_size; ++k)
    {
        if(vIndex[x * max_row_size + k] == y)
        {
            vValue[x * max_row_size + k] += value;
            // printf("%f \n", vValue[x * max_row_size + k]);
                k = max_row_size; // to exit for loop
            }
    }

}



// __global__
// void assembleGrid_GPU(
//     std::size_t N,               // number of elements per row
//     std::size_t dim,             // dimension
//     double* l_value,        // local element's ELLPACK value vector
//     std::size_t* l_index,        // local element's ELLPACK index vector
//     std::size_t l_max_row_size,  // local element's ELLPACK maximum row size
//     std::size_t l_num_rows,      // local element's ELLPACK number of rows
//     double* g_value,        // global element's ELLPACK value vector
//     std::size_t* g_index,        // global element's ELLPACK index vector
//     std::size_t g_max_row_size,  // global element's ELLPACK maximum row size
//     std::size_t g_num_rows,      // global element's ELLPACK number of rows
//     std::size_t* node_index      // vector that contains the corresponding global indices of the node's local indices
// )        
// {
//     int id = threadIdx.x + blockIdx.x*blockDim.x;
    
//     // printf("%d \n", i/2)    ;
//     for ( int i = 0; i < 8; i++ )
//         setAt( 2*node_index[ id/2 ] + ( id % 2 ), 2*node_index[i/2] + ( i % 2 ), g_value, g_index, g_max_row_size, valueAt( 2*(id/2) + ( id % 2 ), i, l_value, l_index, l_max_row_size) );
    
// }


__global__
void assembleGrid2D_GPU(
    std::size_t N,               // number of elements per row
    std::size_t dim,             // dimension
    double* l_value,        // local element's ELLPACK value vector
    std::size_t* l_index,        // local element's ELLPACK index vector
    std::size_t l_max_row_size,  // local element's ELLPACK maximum row size
    std::size_t l_num_rows,      // local element's ELLPACK number of rows
    double* g_value,        // global element's ELLPACK value vector
    std::size_t* g_index,        // global element's ELLPACK index vector
    std::size_t g_max_row_size,  // global element's ELLPACK maximum row size
    std::size_t g_num_rows,      // global element's ELLPACK number of rows
    std::size_t* node_index      // vector that contains the corresponding global indices of the node's local indices
)        
{
    int idx = threadIdx.x + blockIdx.x*blockDim.x;
    int idy = threadIdx.y + blockIdx.y*blockDim.y;

    setAt( 2*node_index[ idx/2 ] + ( idx % 2 ), 2*node_index[idy/2] + ( idy % 2 ), g_value, g_index, g_max_row_size, valueAt( 2*(idx/2) + ( idx % 2 ), 2*(idy/2) + ( idy % 2 ), l_value, l_index, l_max_row_size) );

}

__global__
void transformToELL_GPU(double *array, double *value, size_t *index, size_t max_row_size, size_t num_rows)
{

    int id = blockDim.x * blockIdx.x + threadIdx.x;

    if ( id < num_rows )
    {
        size_t counter = id*max_row_size;
        size_t nnz = 0;
        
			// printf("array = %e\n", array [ 1 ]);
        for ( int j = 0 ; nnz < max_row_size ; j++ )
        {
            if ( array [ j + id*num_rows ] != 0 )
            {
				// printf("array = %e\n", array [ j + id*num_rows ]);
                value [counter] = array [ j + id*num_rows ];
                index [counter] = j;
				// printf("value = %e\n", value[counter]);
                counter++;
                nnz++;
            }
            
            if ( j == num_rows - 1 )
            {
                for ( int i = counter ; nnz < max_row_size ; counter++ && nnz++ )
                {
                    value [counter] = 0.0;
                    index [counter] = num_rows;
                }
            }
        }
    }
}


// calculates the coordinates of each node
// __host__ 
void calculateNodeCoordinates(Node* node, std::size_t numNodes, std::size_t numNodesPerDim, double h)
{
    int ycount = 0;
    for ( int i = 0 ; i < numNodes ; i += numNodesPerDim )
        {
            int count = 0;
 
            for ( int j = i ; j < numNodesPerDim + i ; j++ )
            {
                if ( j == i )
                {
                    node[j].setXCoor( 0.0 );
                    node[j].setYCoor( ycount*h);
                    count++;
                }
                
                else
                {
                    node[j].setXCoor( h*count );
                    node[j].setYCoor( ycount*h);
                    count++;
                }
            }
            ycount++;
        }
}






#endif // TDO_H