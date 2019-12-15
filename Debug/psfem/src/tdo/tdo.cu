/*
    
*/


#include <iostream>
#include <cuda.h>
#include <vector>
#include <cuda_runtime.h>
#include "../include/mycudaheader.h"

using namespace std;


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





__global__
void convertToELL(double *coo, double *ell_val, size_t *ell_ind, size_t max_row_size, size_t N)
{
	unsigned int id = threadIdx.x + blockIdx.x*blockDim.x; // 0 - 5
    unsigned int stride = gridDim.x*blockDim.x; // 6
    
    extern __shared__ size_t temp[];
    
        
        // Inputting the values into ELLPACK index vector

        
        // fill in temp vector with values of COO's each row

        for ( int i = 0 ; i < N ; ++i )
        {
            if(id == 0)
                printf("id*stride + i = %d\n", id*stride + i);

            if ( coo[ id*stride + i ] == 0 )    // TODO: replace 0.0 with something else
                temp[ id*stride + i ] = N;

            else
                temp[ id*stride + i ] = i;
        }
       
        int ind_counter = 0;
        for ( int i = 0 ; i < N || ind_counter == max_row_size ; i++ )
        {
            if ( temp[ id*stride + i ] != N )
            {
                temp[ id*stride + ind_counter ] = temp[ id*stride + i ];
                // temp[ id*stride + i ] = N;
                ind_counter++;
            }
            
            else{}
        }
        

             
    size_t counter = 0;

    // NOTE: potentially, include arranging the temp[] or ell_ind() here as well?
    // Compressing the COO, so that the NNZ are located in the first columns
    for ( int i = 0 ; i < stride ; i++ )
    {
        if ( coo[ i + stride*id ] != 0 )    // TODO: change from 0.0 to abs(value) > 0
        {
            coo[counter + stride*id] = coo[ i + stride*id ];

            if ( counter + stride*id != i + stride*id )
                coo[ i + stride*id ] = 0;

            // coo[ i + stride*id ] = 0;
            counter++;
        }
        
        else{}
    }

    // Inputting the values into ELLPACK value vector
    if ( id < max_row_size )
    {
        for ( int i = 0 ; i < N ; ++i )
        {
            ell_val[ id*stride + i ] = coo [ id + stride*i ];
            ell_ind[ id*stride + i ] = temp [ id + stride*i ];
        }
    }



}


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
        cout << "node [" << m_index << "] = ( " << m_coo[0] << ", " << m_coo[1] << " )" << endl;
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
    Element(size_t ind) : m_index(ind)  // TODO: change int -> size_t
    {   

            m_vValue.resize(25);
            m_vIndex.resize(25);
            
            m_vValue = {4, 	1, 	0, 	1, 	4, 	1, 	1, 	4, 	1, 	1, 	4, 	1, 	1, 	4, 	1, 	1, 	4, 	1, 	1, 	4, 	1, 	1, 	4, 	0};
            m_vIndex = {0, 	1, 	8, 	0, 	1, 	2, 	1, 	2, 	3, 	2, 	3, 	4, 	3, 	4, 	5, 	4, 	5, 	6, 	5, 	6, 	7, 	6, 	7, 	8};

            m_max_row_size = 3;
            m_num_rows = 4;

    }
    

    size_t index()
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
        cout << "Element " << m_index << endl;
        for ( int i = 0 ; i < m_node.size() ; ++i )
            m_node[i]->printCoor();
    }

    double* getValueAddress() { return &m_vValue[0]; }
    size_t* getIndexAddress() { return &m_vIndex[0]; }

    size_t* getNodeGlobalIndex() { return &m_node_index_list[0]; }

    size_t max_row_size() { return m_max_row_size; }
    size_t num_rows() { return m_num_rows; }

    int nodeIndex(int i)
    {
        return m_node[i]->index();
    }

    double operator()(size_t x, size_t j) 
    {
        return m_K[x][j];
    }



private:
    std::vector<Node*> m_node;
    size_t m_index;
    size_t m_max_row_size;
    size_t m_num_rows;
    vector<size_t> m_node_index_list;
    double m_rho;

    double m_K[8][8];   // TODO: change 8 to dimension-friendly variable
    vector<double> m_vValue;
    vector<size_t> m_vIndex;


};

// returns value at A(x,y)
__device__
double valueAt(size_t x, size_t y, double* vValue, size_t* vIndex, size_t max_row_size)
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
void setAt( size_t x, size_t y, double* vValue, size_t* vIndex, size_t max_row_size, double value )
{
    for(size_t k = 0; k < max_row_size; ++k)
    {
        if(vIndex[x * max_row_size + k] == y)
        {
            vValue[x * max_row_size + k] += value;
            // printf("%f \n", vValue[x * max_row_size + k]);
                k = max_row_size; // to exit for loop
            }
    }

}



__global__
void assembleGrid_GPU(
    size_t N,               // number of elements per row
    size_t dim,             // dimension
    double* l_value,        // local element's ELLPACK value vector
    size_t* l_index,        // local element's ELLPACK index vector
    size_t l_max_row_size,  // local element's ELLPACK maximum row size
    size_t l_num_rows,      // local element's ELLPACK number of rows
    double* g_value,        // global element's ELLPACK value vector
    size_t* g_index,        // global element's ELLPACK index vector
    size_t g_max_row_size,  // global element's ELLPACK maximum row size
    size_t g_num_rows,      // global element's ELLPACK number of rows
    size_t* node_index      // vector that contains the corresponding global indices of the node's local indices
)        
{
    int id = threadIdx.x + blockIdx.x*blockDim.x;
    
    // printf("%d \n", i/2)    ;
    for ( int i = 0; i < 8; i++ )
        setAt( 2*node_index[ id/2 ] + ( id % 2 ), 2*node_index[i/2] + ( i % 2 ), g_value, g_index, g_max_row_size, valueAt( 2*(id/2) + ( id % 2 ), i, l_value, l_index, l_max_row_size) );
    
}


__global__
void assembleGrid2D_GPU(
    size_t N,               // number of elements per row
    size_t dim,             // dimension
    double* l_value,        // local element's ELLPACK value vector
    size_t* l_index,        // local element's ELLPACK index vector
    size_t l_max_row_size,  // local element's ELLPACK maximum row size
    size_t l_num_rows,      // local element's ELLPACK number of rows
    double* g_value,        // global element's ELLPACK value vector
    size_t* g_index,        // global element's ELLPACK index vector
    size_t g_max_row_size,  // global element's ELLPACK maximum row size
    size_t g_num_rows,      // global element's ELLPACK number of rows
    size_t* node_index      // vector that contains the corresponding global indices of the node's local indices
)        
{
    int idx = threadIdx.x + blockIdx.x*blockDim.x;
    int idy = threadIdx.y + blockIdx.y*blockDim.y;

    setAt( 2*node_index[ idx/2 ] + ( idx % 2 ), 2*node_index[idy/2] + ( idy % 2 ), g_value, g_index, g_max_row_size, valueAt( 2*(idx/2) + ( idx % 2 ), 2*(idy/2) + ( idy % 2 ), l_value, l_index, l_max_row_size) );

}



// calculates the coordinates of each node
__host__
void calculateNodeCoordinates(Node* node, size_t numNodes, size_t numNodesPerDim, double h)
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



int main()
{

    size_t N = 2;
    size_t dim = 2;
    double rho = 0.4;

    // calculate the number of elements in the domain                                                               
    size_t numElements = pow(N,dim);
    size_t numNodesPerDim = N + 1;
    size_t numNodes = numNodesPerDim*numNodesPerDim;

    // calculate h
    float h = 1.0/N;

    // create an array of nodes
    vector<Node> node;
    
    for ( int i = 0 ; i < numNodes ; ++i )
        node.push_back(Node(i));


        calculateNodeCoordinates(&node[0], numNodes, numNodesPerDim, h);
        
        
        // creating an array of elements
        vector<Element> element;
        
        for ( int i = 0 ; i < numElements ; i++ )
        element.push_back( Element(i) );
        
        
        // adding node indices
    for ( int i = 0 ; i < numElements ; i++ )
    {
        element[i].addNode(&node[ i + i/N ]);   // lower left node
        element[i].addNode(&node[ i + i/N + 1]);   // lower right node
        element[i].addNode(&node[ i + i/N + N + 1]);   // upper left node
        element[i].addNode(&node[ i + i/N + N + 2]);   // upper right node
    }
    

    // flattened global matrix
    vector<double> K = {4,	1,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0, \
                        1,	4,	1,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0, \
                        0,	1,	8,	2,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0, \
                        0,	0,	2,	8,	1,	0,	1,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0, \
                        0,	0,	0,	1,	4,	1,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0, \
                        0,	0,	0,	0,	1,	4,	0,	0,	1,	0,	0,	0,	0,	0,	0,	0,	0,	0, \
                        0,	0,	0,	1,	0,	0,	8,	2,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0, \
                        0,	0,	0,	0,	0,	0,	2,	8,	2,	0,	0,	0,	0,	0,	0,	0,	0,	0, \
                        0,	0,	0,	0,	0,	1,	0,	2,	16,	4,	0,	0,	0,	0,	0,	0,	0,	0, \
                        0,	0,	0,	0,	0,	0,	0,	0,	4,	16,	2,	0,	1,	0,	0,	0,	0,	0, \
                        0,	0,	0,	0,	0,	0,	0,	0,	0,	2,	8,	2,	0,	0,	0,	0,	0,	0, \
                        0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	2,	8,	0,	0,	1,	0,	0,	0, \
                        0,	0,	0,	0,	0,	0,	0,	0,	0,	1,	0,	0,	4,	1,	0,	0,	0,	0, \
                        0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	1,	4,	1,	0,	0,	0, \
                        0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	1,	0,	1,	8,	2,	0,	0, \
                        0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	2,	8,	1,	0, \
                        0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	1,	4,	1, \
                        0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	1,	4  };
        

    // CUDA

    // host

    size_t max_row_size = 0;
    double num_rows = 18;

    // device
    double* d_K             = nullptr;
    double* d_K_value       = nullptr;
    size_t* d_K_index       = nullptr;
    size_t* d_max_row_size  = nullptr;
    int* d_mutex            = nullptr;

    CUDA_CALL( cudaMalloc( (void**)&d_K, sizeof(double) * 18 * 18 )     );
    CUDA_CALL( cudaMalloc( (void**)&d_max_row_size, sizeof(size_t) )    );
    CUDA_CALL( cudaMalloc( (void**)&d_mutex, sizeof(int) ) );

    CUDA_CALL( cudaMemset(d_max_row_size, 0, sizeof(size_t)) );
    CUDA_CALL( cudaMemset(d_mutex, 0, sizeof(int)) );
    
    CUDA_CALL( cudaMemcpy(d_K, &K[0], sizeof(double) * 18 * 18 , cudaMemcpyHostToDevice) ); 


    // calculate global matrix's max_row_size
    getMaxRowSize<<< 1 , 18 >>>(d_K, d_max_row_size, d_mutex, 18);
    CUDA_CALL( cudaMemcpy(&max_row_size, d_max_row_size, sizeof(size_t), cudaMemcpyDeviceToHost ) ); 
    
    cout << max_row_size << endl;
    // allocate device memory for global stiffness matrix's ELLPACK value and index vectors
    CUDA_CALL( cudaMalloc( (void**)&d_K_value, sizeof(double) * 18 * max_row_size )     );
    CUDA_CALL( cudaMalloc( (void**)&d_K_index, sizeof(size_t) * 18 * max_row_size )     );
    
    // transform K to ELLPACK
    transformToELL_GPU<<<1, 18>>>(d_K, d_K_value, d_K_index, max_row_size, 18);
    
    
    // deallocate big K matrix, no needed now
    cudaFree( d_K );
    
    
    // copy and allocate the node index of each element
    
    vector<size_t*> d_node_index(numElements);
    
    for ( int i = 0 ; i < numElements ; i++ )
    {
        CUDA_CALL( cudaMalloc( (void**)&d_node_index[i], sizeof(size_t) * 4 )     );
        CUDA_CALL( cudaMemcpy( d_node_index[i], element[i].getNodeGlobalIndex() , sizeof(size_t) * 4 , cudaMemcpyHostToDevice ) ); 
    }


    // obtain k elements' value and index vectors
    // allocate k element stiffness matrices

    vector<double*> d_ke_value(numElements);
    vector<size_t*> d_ke_index(numElements);


    // allocate and copy elements' ELLPACK stiffness matrices to device (value and index vectors)
    for ( int i = 0 ; i < numElements ; i++ )
    {
        CUDA_CALL( cudaMalloc( (void**)&d_ke_value[i], sizeof(double) * 24 )     );
        CUDA_CALL( cudaMalloc( (void**)&d_ke_index[i], sizeof(size_t) * 24 )     );

        CUDA_CALL( cudaMemcpy( d_ke_value[i], element[i].getValueAddress() , sizeof(double) * 24 , cudaMemcpyHostToDevice ) ); 
        CUDA_CALL( cudaMemcpy( d_ke_index[i], element[i].getIndexAddress() , sizeof(size_t) * 24 , cudaMemcpyHostToDevice ) ); 
    }
    


    // array of the initial design variable

    vector<double> design(numElements);

    for ( int i = 0 ; i < numElements ; i++ )
        design.push_back(rho);

    double* d_design = nullptr;
        
    CUDA_CALL( cudaMalloc( (void**)&d_design, sizeof(double) * numElements )     );
    CUDA_CALL( cudaMemcpy( d_design, &design[0] , sizeof(double) * numElements , cudaMemcpyHostToDevice ) ); 
    


    // allocate and copy the empty global matrix

    Element global;
    
    double* d_KG_value;
    size_t* d_KG_index;
    
    CUDA_CALL( cudaMalloc( (void**)&d_KG_value, sizeof(double) * 72 )     );
    CUDA_CALL( cudaMalloc( (void**)&d_KG_index, sizeof(size_t) * 72 )     );
    CUDA_CALL( cudaMemcpy( d_KG_value, global.getValueAddress() , sizeof(double) * 72 , cudaMemcpyHostToDevice ) ); 
    CUDA_CALL( cudaMemcpy( d_KG_index, global.getIndexAddress() , sizeof(size_t) * 72 , cudaMemcpyHostToDevice ) ); 
    

    // add local stiffness matrices into the global

    // for ( int i = 0 ; i < numElements ; i++ )
    // {
    //     assembleGrid_GPU<<<1, 8>>>( 2, 2, d_ke_value[i], d_ke_index[i], element[i].max_row_size(), element[i].num_rows(), d_KG_value, d_K_index, global.max_row_size(), global.num_rows(), d_node_index[i] );
    //     cudaDeviceSynchronize();
    // }

    dim3 blockDim(8,8,1);
    //     assembleGrid2D_GPU<<<1, blockDim>>>( 2, 2, d_ke_value[0], d_ke_index[0], element[0].max_row_size(), element[0].num_rows(), d_KG_value, d_K_index, global.max_row_size(), global.num_rows(), d_node_index[0] );

    for ( int i = 0 ; i < numElements ; i++ )
        {
            assembleGrid2D_GPU<<<1, blockDim>>>( 2, 2, d_ke_value[i], d_ke_index[i], element[i].max_row_size(), element[i].num_rows(), d_KG_value, d_K_index, global.max_row_size(), global.num_rows(), d_node_index[i] );
            cudaDeviceSynchronize();
        }



    printVector_GPU<<<1,72>>> ( d_KG_value, 72 );
    // printVector_GPU<<<1,72>>> ( d_K_index, 72 );
    cudaDeviceSynchronize();

}