/*
NOTE:

    pseudocode:

    insert N, which is the number of elements per row
    calculate each element's local stiffness matrix (include the design variable (TDO's density))
    assemble a global stiffness matrix from N, in COO
    convert COO to ELLPACK

    solve for u

    



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
    Element(size_t ind)
    {   
        m_index = ind; 
    
        // DEBUG: testing, playing around with stiffness matrices

        for ( int i = 0 ; i < 8 ; i++ )
        {
            for ( int j = 0 ; j < 8 ; j++ )
            {
                if ( i == j )
                    m_K[i][j] = 4.0;

                else
                    m_K[i][j] = 1.0;


            }
            
        }
        
                

    }

    size_t index()
    {
        return m_index;
    }

    void addNode(Node *x)
    {
        m_node.push_back(x);
    }

    void printNodes()
    {
        cout << "Element " << m_index << endl;
        for ( int i = 0 ; i < m_node.size() ; ++i )
            m_node[i]->printCoor();
    }

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

    double m_K[8][8];   // TODO: change 8 to dimension-friendly variable


};


void assembleGrid(size_t N, size_t dim, vector<Element> &element, vector<Node> &node, double *flat_K)
{
    size_t numElements = pow(N,dim);
    size_t numNodesPerDim = N + 1;

    // adding the first node
    for ( int i = 0 ; i < numElements ; i++ )
    {
        element[i].addNode(&node[ i + i/N ]);   // lower left node
        element[i].addNode(&node[ i + i/N + 1]);   // lower right node
        element[i].addNode(&node[ i + i/N + N + 1]);   // upper left node
        element[i].addNode(&node[ i + i/N + N + 2]);   // upper right node
    }

    double K_Global[ 18 ][ 18 ];    // TODO: dim

    for ( int i = 0 ; i < 18 ; i++)
    {
        for ( int j = 0 ; j < 18 ; j++)
            {
                K_Global[i][j] = 0;
            }

    }

    cout << "" << endl;

    for ( int elmn_index = 0 ; elmn_index < numElements ; elmn_index++ )
    {
        for ( int x = 0 ; x < 4 ; x++ ) // TODO: dim  
        {
            for ( int y = 0 ; y < 4 ; y++ )        // TODO: dim   
            {
                K_Global[ 2*element[elmn_index].nodeIndex(x)     ][ 2*element[elmn_index].nodeIndex(y)     ] += element[0]( 2*x    , 2*y          );
                K_Global[ 2*element[elmn_index].nodeIndex(x)     ][ 2*element[elmn_index].nodeIndex(y) + 1 ] += element[0]( 2*x    , 2*y + 1      );
                K_Global[ 2*element[elmn_index].nodeIndex(x) + 1 ][ 2*element[elmn_index].nodeIndex(y)     ] += element[0]( 2*x + 1, 2*y          );
                K_Global[ 2*element[elmn_index].nodeIndex(x) + 1 ][ 2*element[elmn_index].nodeIndex(y) + 1 ] += element[0]( 2*x + 1, 2*y + 1      );
            }
        }
    }
   

    // DEBUG: matrix output
        // for ( int i = 0 ; i < 18 ; i++)
        // {
        //     for ( int j = 0 ; j < 18 ; j++)
        //         {
        //             cout << K_Global[i][j] << " ";
        //         }
        //     cout << "\n";
        // }

    // flatten matrix

    for ( int i = 0 ; i < 18 ; i++)
        {
            for ( int j = 0 ; j < 18 ; j++)
                flat_K[i*18 + j] = K_Global[i][j];
        }

    
}


int main()
{

    size_t N = 2;
    size_t dim = 2;

    // calculate the number of elements in the domain                                                               

    size_t numElements = pow(N,dim);
    size_t numNodesPerDim = N + 1;
    size_t numNodes = numNodesPerDim*numNodesPerDim;

    // calculate h
    float h = 1.0/N;

    // create the nodes
    
    vector<Node> node;
    
    for ( int i = 0 ; i < numNodes ; ++i )
    {
        node.push_back(Node(i));
    }

    // calculate each node's coordinates
    
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

        // for ( int i = 0 ; i < numNodes ; ++i )
        // {
        //     node[i].printCoor();
        // }

    // // deallocation

    // CUDA_CALL( cudaFree(d_A_coo)            );
    // CUDA_CALL( cudaFree(d_max_row_size)    );


    // creating elements

    vector<Element> element;

    for ( int i = 0 ; i < numElements ; i++ )
        element.push_back( Element(i) );


    vector<double> flat_K_Global(18*18);

    assembleGrid(N, dim, element, node, &flat_K_Global[0]);
    
    for ( int i = 0 ; i < 10 ; ++i )
        {
            cout << flat_K_Global[i] << endl;
        }


}