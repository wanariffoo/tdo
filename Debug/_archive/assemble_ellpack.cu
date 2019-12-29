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

        // for ( int i = 0 ; i < 8 ; i++ )
        // {
        //     for ( int j = 0 ; j < 8 ; j++ )
        //     {
        //         if ( i == j )
        //             m_K[i][j] = 4.0;

        //         else
        //             m_K[i][j] = 1.0;

        //     }
            
        // }

    }

    size_t index()
    {
        return m_index;
    }

    double* valueAddress()
    {
        return &m_vValue[0];
    }

    double* indexAddress()
    {
        return &m_vIndex[0];
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

    // double operator()(size_t x, size_t j) 
    // {
    //     return m_K[x][j];
    // }

    double operator()(size_t x, size_t y) 
    {
        for(size_t k = 0; k < m_max_row_size; ++k){
            if(m_vIndex[x * m_max_row_size + k] == y)
               {
                    return m_vValue[x * m_max_row_size + k];
               }
        }
        return 0.0;
    }

private:
    std::vector<Node*> m_node;
    size_t m_index;

    double m_K[8][8];   // TODO: change 8 to dimension-friendly variable

    vector<double> m_vValue = {4,1,	0,	1,	4,	1,	1,	4,	1,	1,	4,	1,	1,	4,	1,	1,	4,	1,	1,	4,	1,	1,	4,	0   };
    // vector<double> m_vValue = {4,	1,	0,	1,	4,	1,	1,	4,	1,	1,	4,	1,	1,	4,	1,	1,	4,	1,	1,	4,	1,	1,	4,	0};
    vector<double> m_vIndex = {0,	1,	8,	0,	1,	2,	1,	2,	3,	2,	3,	4,	3,	4,	5,	4,	5,	6,	5,	6,	7,	6,	7,	8};
    


    // vector<double> m_vIndex = {0,	0,	1,	2,	3,	4,	5,	6,	1,	1,	2,	3,	4,	5,	6,	7,	8,	2,	3,	4,	5,	6,	7,	8};

    size_t m_max_row_size = 3;
};



class ElementGlobal
{
public:
    ElementGlobal()
    {   
        m_vValue.resize(125);
        m_vIndex.resize(125);

        for ( int i = 0 ; i < 125 ; i++)
        {
            m_vValue[i] = 0.0000000001;
        }
        
        // m_vValue = { 4,	1,	1,	2,	1,	1,	1,	2,	1,	4,	2,	2,	1,	1,	1,	4,	1,	1,	1,	4,	8,	8,	4,	4,	8,	8,	2,	16,	8,	8,	4,	4,	1,	1,	4,	5,	0,	1,	2,	1,	1,	5,	2,	2,	12,	2,	2,	1,	1,	1,	8,	2,	5,	5,	0,	0,	0,	1,	0,	1,	1,	1,	3,	1,	0,	0,	0,	0,	2,	8,	1,	1,	0,	0,	0,	0,	0,	0,	4,	4,	1,	0,	0,	0,	0,	0,	0,	1,	1,	1,	0,	0,	0,	0,	0,	0,	1,	1,	4,	0,	0,	0,	0,	0,	0,	0,	4,	4,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	1,	0};
        
        m_vIndex = {0,	1,	18,	18,	18,	18,	18,	0,	1,	2,	18,	18,	18,	18,	1,	2,	3,	18,	18,	18,	18,	2,	3,	4,	6,	18,	18,	18,	3,	4,	5,	18,	18,	18,	18,	4,	5,	8,	9,	18,	18,	18,	3,	6,	7,	8,	9,	10,	18,	6,	7,	8,	9,	10,	11,	18,	5,	7,	8,	9,	10,	11,	18,	8,	9,	10,	12,	18,	18,	18,	9,	10,	11,	18,	18,	18,	18,	10,	11,	14,	18,	18,	18,	18,	9,	12,	13,	18,	18,	18,	18,	12,	13,	14,	18,	18,	18,	18,	11,	13,	14,	15,	18,	18,	18,	0,	1,	14,	15,	16,	18,	18,	0,	1,	2,	3,	15,	16,	17,	1,	2,	3,	4,	16,	17,	18};
        

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

    double* valueAddress()
    {
        return &m_vValue[0];
    }

    double* indexAddress()
    {
        return &m_vIndex[0];
    }

    // double operator()(size_t x, size_t j) 
    // {
    //     return m_K[x][j];
    // }

    double operator()(size_t x, size_t y) 
    {
        for(size_t k = 0; k < m_max_row_size; ++k){
            
            if(m_vIndex[x * m_max_row_size + k] == y)
                return m_vValue[x * m_max_row_size + k];
        }
        return 0.0;
    }

    double test(size_t x, size_t y) 
    {
        return m_vValue[0];
        
    }

    void set(double value, size_t x, size_t y)
    {
        for(size_t k = 0; k < m_max_row_size; ++k){
            if(m_vIndex[x * m_max_row_size + k] == y)
            {
                if ( m_vValue[x * m_max_row_size + k] == 0 ) { }
                
                else 
                {
                    m_vValue[x * m_max_row_size + k] += value;
                }
            }
        }
    }

private:
    std::vector<Node*> m_node;
    size_t m_index;

    double m_K[8][8];   // TODO: change 8 to dimension-friendly variable

    vector<double> m_vValue; // {4,1,1,2,1,1,1,4,8,8,4,4,0,1,2,1,1,0}
    vector<double> m_vIndex; // {0,0,1,2,3,4,1,1,2,3,4,5,6,2,3,4,5,6};

    size_t m_max_row_size = 7;


};

void assembleGrid(size_t N, size_t dim, vector<Element> &element, vector<Node> &node, ElementGlobal &K_Global)
{
    size_t numElements = pow(N,dim);
    size_t numNodesPerDim = N + 1;

    // adding node indices
    for ( int i = 0 ; i < numElements ; i++ )
    {
        element[i].addNode(&node[ i + i/N ]);   // lower left node
        element[i].addNode(&node[ i + i/N + 1]);   // lower right node
        element[i].addNode(&node[ i + i/N + N + 1]);   // upper left node
        element[i].addNode(&node[ i + i/N + N + 2]);   // upper right node
    }

    

    // K_Global.set(element[0]( 0, 0), 0, 0);
    // K_Global.set(element[0]( 0, 1), 0, 1);
    // K_Global.set(element[0]( 1, 0), 1, 0);
    // K_Global.set(element[0]( 1, 1), 1, 1);


    
    // cout << K_Global.test(0,0) << endl;

    // cout << "" << endl;

    for ( int elmn_index = 0 ; elmn_index < numElements ; elmn_index++ )
    {
        for ( int x = 0 ; x < 4 ; x++ ) // TODO: dim  
        {
            for ( int y = 0 ; y < 4 ; y++ )        // TODO: dim   
            {       
                    // set ( value, row, col )
                    // if ( element[elmn_index]( 2*0    , 2*0          ) != 0 )
                    K_Global.set(element[elmn_index]( 2*x    , 2*y          ), 2*element[elmn_index].nodeIndex(x)    , 2*element[elmn_index].nodeIndex(y)       );

                    // if ( element[elmn_index]( 2*0    , 2*0 + 1      ) != 0 )
                    K_Global.set(element[elmn_index]( 2*x    , 2*y + 1      ), 2*element[elmn_index].nodeIndex(x)    , 2*element[elmn_index].nodeIndex(y) + 1   );

                    // if ( element[elmn_index]( 2*0 + 1, 2*0          ) != 0 )
                    K_Global.set(element[elmn_index]( 2*x + 1, 2*y          ), 2*element[elmn_index].nodeIndex(x) + 1, 2*element[elmn_index].nodeIndex(y)       );

                    // if ( element[elmn_index]( 2*0 + 1, 2*0 + 1      ) != 0 )
                    K_Global.set(element[elmn_index]( 2*x + 1, 2*y + 1      ), 2*element[elmn_index].nodeIndex(x) + 1, 2*element[elmn_index].nodeIndex(y) + 1   );
    
                    // K_Global.set(element[elmn_index]( 2*x    , 2*y          ), 2*element[elmn_index].nodeIndex(x)    , 2*element[elmn_index].nodeIndex(y)       );
                    // K_Global.set(element[elmn_index]( 2*x    , 2*y + 1      ), 2*element[elmn_index].nodeIndex(x)    , 2*element[elmn_index].nodeIndex(y) + 1   );
                    // K_Global.set(element[elmn_index]( 2*x + 1, 2*y          ), 2*element[elmn_index].nodeIndex(x) + 1, 2*element[elmn_index].nodeIndex(y)       );
                    // K_Global.set(element[elmn_index]( 2*x + 1, 2*y + 1      ), 2*element[elmn_index].nodeIndex(x) + 1, 2*element[elmn_index].nodeIndex(y) + 1   );
    
    
                    // K_Global[ 2*element[elmn_index].nodeIndex(x)     ][ 2*element[elmn_index].nodeIndex(y)     ] += element[0]( 2*x    , 2*y          );
                    // K_Global[ 2*element[elmn_index].nodeIndex(x)     ][ 2*element[elmn_index].nodeIndex(y) + 1 ] += element[0]( 2*x    , 2*y + 1      );
                    // K_Global[ 2*element[elmn_index].nodeIndex(x) + 1 ][ 2*element[elmn_index].nodeIndex(y)     ] += element[0]( 2*x + 1, 2*y          );
                    // K_Global[ 2*element[elmn_index].nodeIndex(x) + 1 ][ 2*element[elmn_index].nodeIndex(y) + 1 ] += element[0]( 2*x + 1, 2*y + 1      );
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

    // for ( int i = 0 ; i < 18 ; i++)
    //     {
    //         for ( int j = 0 ; j < 18 ; j++)
    //             flat_K[i*18 + j] = K_Global[i][j];
    //     }

    
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

        vector<Element> element;

        
        for ( int i = 0 ; i < numElements ; i++ )
        element.push_back( Element(i) );

        ElementGlobal K_Global;

        // K_Global.set(4,0,0);
        
        // vector<double> flat_K_Global(18*18);
        
    
        // for ( int i = 0 ; i < 8 ; i++ )
        // {
        //     for ( int j = 0 ; j < 8 ; j++ )
        //         {
        //             // cout << "k[0] (" << i << ", " << j << ") : " << element[0](i,j) << endl;
        //             cout << element[0](i,j) << " ";
        //         }    
        //     cout << "" << endl;
        // }
        
        cout << K_Global(8,6) << endl;
        
        assembleGrid(N, dim, element, node, K_Global);
        
        // cout << element[0](0,1) << endl;
        // K_Global.set(3,1,0);
        
        
        cout << "Global matrix : \n";
        for ( int i = 0 ; i < 18 ; i++ )
        {
            for ( int j = 0 ; j < 18 ; j++ )
                cout << K_Global(i,j) << " ";

            cout << "\n";

        }
}