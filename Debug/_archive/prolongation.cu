#include <iostream>
#include <vector>
#include <cmath>

using namespace std;



// void AssembleProlMatrix(vector<vector<vector<double>>> &P, vector<vector<size_t>> N, vector<size_t> numNodes, vector<size_t> num_rows,  )

int main()
{

    vector<vector<size_t>> N;
    N.resize(2, vector<size_t>(2));
    N[0] = {3, 1};
    N[1] = {6, 2};

    size_t lev = 0;
    size_t dim = 2;
    vector<size_t> numNodes(2);

    numNodes[0] = 8;
    numNodes[1] = 21;

    vector<size_t> num_rows = { 16, 42 };
    // vector<size_t> num_rows = { numNodes[0]*2, numNodes[1]*2 };

    vector<vector<vector<double>>> A;
    A.resize(2);
    A[0].resize(num_rows[0], vector<double>(num_rows[0]));
    A[1].resize(num_rows[1], vector<double>(num_rows[1]));

    size_t numLevels = 2;
    vector<vector<vector<double>>> P;

    P.resize(numLevels - 1);
    P[0].resize(num_rows[1], vector<double>(num_rows[0]));


    // assembleProlMatrix()


    for ( int i = 0 ; i < numNodes[0]*2 ; i += 2)
    {
        for ( int j = 0 ; j < dim ; j++ )
        {
        
        // same node
            P[lev][( 2*(i % ( (N[lev][0] + 1)*dim) )) + ( (ceil)( i / ( 2*(N[lev][0] + 1 ) ) ) )*2*dim*(N[lev+1][0] + 1) + j][i+j] = 1;

        // east node
        if ( (i / 2 + 1) % (N[lev][0]+1) != 0 )
            P[lev][( 2*(i % ( (N[lev][0] + 1)*dim) )) + ( (ceil)( i / ( 2*(N[lev][0] + 1 ) ) ) )*2*dim*(N[lev+1][0] + 1) + j + 2][i+j] += 0.5;

        // north node
        if ( i / 2 + (N[lev][0] + 1) < (N[lev][0] + 1)*(N[lev][1] + 1))
            P[lev][( 2*(i % ( (N[lev][0] + 1)*dim) )) + ( (ceil)( i / ( 2*(N[lev][0] + 1 ) ) ) )*2*dim*(N[lev+1][0] + 1) + j + 2*(N[lev+1][0] + 1) ][i+j] += 0.5;

        // west node
        if ( (i / 2) % (N[lev][0]+1) != 0 )
            P[lev][( 2*(i % ( (N[lev][0] + 1)*dim) )) + ( (ceil)( i / ( 2*(N[lev][0] + 1 ) ) ) )*2*dim*(N[lev+1][0] + 1) + j - 2][i+j] += 0.5;

        // south node
        if ( i / 2 >= N[lev][0] + 1)
            P[lev][( 2*(i % ( (N[lev][0] + 1)*dim) )) + ( (ceil)( i / ( 2*(N[lev][0] + 1 ) ) ) )*2*dim*(N[lev+1][0] + 1) + j - 2*(N[lev+1][0] + 1)][i+j] += 0.5;

        // north-east node
        if ( (i / 2 + 1) % (N[lev][0]+1) != 0 && i / 2 + (N[lev][0] + 1) < (N[lev][0] + 1)*(N[lev][1] + 1))
            P[lev][( 2*(i % ( (N[lev][0] + 1)*dim) )) + ( (ceil)( i / ( 2*(N[lev][0] + 1 ) ) ) )*2*dim*(N[lev+1][0] + 1) + j + 2*(N[lev+1][0] + 1) + 2 ][i+j] = 0.25;

        // north-west node
        if ( i / 2 + (N[lev][0] + 1) < (N[lev][0] + 1)*(N[lev][1] + 1) && (i / 2) % (N[lev][0]+1) != 0 )
            P[lev][( 2*(i % ( (N[lev][0] + 1)*dim) )) + ( (ceil)( i / ( 2*(N[lev][0] + 1 ) ) ) )*2*dim*(N[lev+1][0] + 1) + j + 2*(N[lev+1][0] + 1) - 2 ][i+j] = 0.25;

        // south-east node
        if ( i / 2 >= N[lev][0] + 1 && (i / 2 + 1) % (N[lev][0]+1) != 0 )
            P[lev][( 2*(i % ( (N[lev][0] + 1)*dim) )) + ( (ceil)( i / ( 2*(N[lev][0] + 1 ) ) ) )*2*dim*(N[lev+1][0] + 1) + j - 2*(N[lev+1][0] + 1) + 2 ][i+j] = 0.25;

        // south-west node
        if ( i / 2 >= N[lev][0] + 1 && (i / 2) % (N[lev][0]+1) != 0 )
            P[lev][( 2*(i % ( (N[lev][0] + 1)*dim) )) + ( (ceil)( i / ( 2*(N[lev][0] + 1 ) ) ) )*2*dim*(N[lev+1][0] + 1) + j - 2*(N[lev+1][0] + 1) - 2 ][i+j] = 0.25;


        }
    }

    



    for ( int i = 0 ; i < num_rows[1] ; i++ )
    {
        for ( int j = 0 ; j < num_rows[0] ; j++ )
            cout << P[0][i][j] << " ";

        cout << "\n";
    }

}
