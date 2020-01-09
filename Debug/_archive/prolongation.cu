#include <iostream>
#include <vector>
#include <cmath>

using namespace std;





int main()
{

    vector<vector<size_t>> N;
    N.resize(2, vector<size_t>(2));
    N[0] = {3, 1};
    N[1] = {6, 2};

    vector<size_t> num_rows = { 16, 42 };


    vector<vector<vector<double>>> A;
    A.resize(2);
    A[0].resize(num_rows[0], vector<double>(num_rows[0]));
    A[1].resize(num_rows[1], vector<double>(num_rows[1]));

    size_t numLevels = 2;
    vector<vector<vector<size_t>>> P;

    P.resize(numLevels - 1);
    P[0].resize(num_rows[1], vector<size_t>(num_rows[0]));



    size_t lev = 0;
    size_t dim = 2;
    vector<size_t> numNodes(2);

    numNodes[0] = 8;
    numNodes[1] = 21;

    // for (  int j = 0; j < 12 ; j++ )
    // {

    // }

    // int j = 0;
    // P[lev][( 2*(j % ( (N[lev][0] + 1)*dim) )) + ( (j/(2*(N[lev][0] + 1))))*20][j] = 1;
    // P[lev][( 2*(j % ( (N[lev][0] + 1)*dim) )) + ( (j/(2*(N[lev][0] + 1))))*20+1][j+1] = 1;

    // j = 2;
    // P[lev][( 2*(j % ( (N[lev][0] + 1)*dim) )) + ( (j/(2*(N[lev][0] + 1))))*20][j] = 1;
    // P[lev][( 2*(j % ( (N[lev][0] + 1)*dim) )) + ( (j/(2*(N[lev][0] + 1))))*20+1][j+1] = 1;

    // j = 4;
    // P[lev][( 2*(j % ( (N[lev][0] + 1)*dim) )) + ( (j/(2*(N[lev][0] + 1))))*20][j] = 1;
    // P[lev][( 2*(j % ( (N[lev][0] + 1)*dim) )) + ( (j/(2*(N[lev][0] + 1))))*20+1][j+1] = 1;

    // j = 6;
    // P[lev][( 2*(j % ( (N[lev][0] + 1)*dim) )) + ( (j/(2*(N[lev][0] + 1))))*20][j] = 1;
    // P[lev][( 2*(j % ( (N[lev][0] + 1)*dim) )) + ( (j/(2*(N[lev][0] + 1))))*20+1][j+1] = 1;


    // for ( int i = 0 ; i < numNodes[0]*2 ; i += 2)
    for ( int i = 0 ; i < numNodes[0] * 2 ; i += 2)
    {
        for ( int j = 0 ; j < dim ; j++ )
        {
            P[lev][( 2*(i % ( (N[lev][0] + 1)*dim) )) + ( (i/(2*(N[lev][0] + 1))))*2*dim*(N[lev+1][0] + 1) + j][i+j] = 1;
        }
    }

    


    // cout << 0 % 6 << endl;
    // cout << 2 % 6 << endl;
    // cout << 4 % 6 << endl;


    for ( int i = 0 ; i < num_rows[1] ; i++ )
    {
        for ( int j = 0 ; j < num_rows[0] ; j++ )
            cout << P[0][i][j] << " ";

        cout << "\n";
    }

}
