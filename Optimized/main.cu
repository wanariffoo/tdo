/*
    GPU-accelerated Thermodynamic Topology Optimization
*/

#include <iostream>
#include <vector>
#include "include/cudakernels.h"
#include "include/assemble.h"
#include "include/solver.h"
#include "include/tdo.h"
#include "include/vtk.h"

using namespace std;


// TODO: store local k matrix in constant/texture memory
// TODO: __device__ valueAt() has x and y mixed up
// NOTE:CHECK: when using shared memory, more than one block, get this error : CUDA error for cudaMemcpy( ...)
// TODO: check that all kernels have (row, col) formats
// TODO: h = diagonal length of the quad
// TODO: have cudamemcpy for prol and rest matrices to be outside of the function
// TODO: deallocation

// URGENT !!!
// TODO: TODO: size_t is used instead of int in matrix assembly (fillProl ...), shouldn't be used as size_t can't contain negative values!!

//// PARALELLIZABLE / OPTIMIZATION
// TODO: fillIndexVector_GPU()
// TODO: shared memory, use 8 bytes for double precision to avoid bank conflict
            // cudaDeviceSetSharedMemConfig( cudaSharedMemBankSizeEightByte )
            // see notes in compendium

//// LOW PRIORITY
// TODO: VTK class
// TODO: RA and AP's valueAt(indices) are a bit messed up and confusing
// TODO: 3d assembly, add for loop
// TODO: tried to fix ApplyTranspose for r_coarse to Apply using R-matrix, but didn't work. using ApplyTranspose for now
    // not much time difference though, from 0.01 to 0.007
    

int main()
{
  
    // create vtk files
    bool writeToVTK = true;

    // material properties
    double youngMod = 200000;
    double poisson = 0.33;

    //// model set-up
    size_t numLevels = 3;
    
    vector<size_t> N;
    vector<vector<size_t>> bc_index(numLevels);



    // // CASE 0 : 2D MBB
    // N = {3,1};                  // domain dimension (x,y,z) on coarsest grid
    // double h_coarse = 1;        // local element mesh size on coarsest grid
    // size_t bc_case = 0;
    // double damp = 2.0/3.0;      // smoother (jacobi damping parameter)




    // CASE 1 : 3D MBB
    N = {6,2,1};                // domain dimension (x,y,z) on coarsest grid
    double h_coarse = 0.5;      // local element mesh size on coarsest grid
    size_t bc_case = 1;
    double damp = 1.0/3.0;      // smoother (jacobi damping parameter)





    
    // applying boundary conditions
    size_t dim = N.size();
    bc_index = applyBC(N, numLevels, bc_case, dim);

    // calculating the mesh size on the top level grid
    double h = h_coarse/pow(2,numLevels - 1);
    size_t local_num_rows = pow(2,dim)*dim;

    // TDO
    double rho = 0.3;
    size_t p = 3;
    double etastar = 12.0;
    double betastar = 2 * pow(h,2);

    vector<size_t> num_rows;
    vector<size_t> max_row_size;
    vector<size_t> p_max_row_size;
    vector<size_t> r_max_row_size;

    //// device pointers
    // local stiffness
    double* d_A_local;

    // global stiffness matrix on each grid-level
    vector<double*> d_value;
    vector<size_t*> d_index;

    // prolongation matrices
    vector<double*> d_p_value;
    vector<size_t*> d_p_index;

    // restriction matrices
    vector<double*> d_r_value;
    vector<size_t*> d_r_index;

    // design variable
    double* d_chi;



    //// CUDA
    vector<size_t*> d_node_index;

    /* ##################################################################
    #                           ASSEMBLY                                #
    ###################################################################*/
       

    Assembler Assembly(dim, h, N, youngMod, poisson, rho, p, numLevels);
    Assembly.setBC(bc_index);
    Assembly.init_GPU(d_A_local, d_value, d_index, d_p_value, d_p_index, d_r_value, d_r_index, d_chi, num_rows, max_row_size, p_max_row_size, r_max_row_size, d_node_index);

    cout << "### GPU-accelerated Thermodynamic Topology Optimization ###" << endl;
    cout << "Dimension: " << dim << endl;
    cout << "Number of Multigrid Levels: " << numLevels << endl;
    cout << "Top-level grid size = { " << Assembly.getGridSize()[0];
   
        for ( int i = 1 ; i < dim ; ++i )
            cout << ", " << Assembly.getGridSize()[i];
        
    cout << " }" << endl;
    
    cout << "Top-level mesh size = " << h << endl;
    cout << "Top-level number of rows = " << num_rows[numLevels - 1] << endl;
    cout << "Number of Elements = " << Assembly.getNumElements() << endl;
    cout << "Assembly ... DONE" << endl;
  
    // load vector, b
    vector<double> b(num_rows[numLevels - 1], 0);
    double force = -1;
    applyLoad(b, N, numLevels, bc_case, dim, force);

    


    double* d_u;
    double* d_b;
    CUDA_CALL( cudaMalloc((void**)&d_u, sizeof(double) * num_rows[numLevels - 1] ) );
    CUDA_CALL( cudaMalloc((void**)&d_b, sizeof(double) * num_rows[numLevels - 1] ) );

    CUDA_CALL( cudaMemset(d_u, 0, sizeof(double) * num_rows[numLevels - 1]) );
    CUDA_CALL( cudaMemcpy(d_b, &b[0], sizeof(double) * num_rows[numLevels - 1], cudaMemcpyHostToDevice) );



    /* ##################################################################
    #                           SOLVER                                  #
    ###################################################################*/

    Solver GMG(d_value, d_index, max_row_size, d_p_value, d_p_index, p_max_row_size, d_r_value, d_r_index, r_max_row_size, numLevels, num_rows, damp);
    
    
    GMG.set_convergence_params(100000, 1e-99, 1e-12);
    GMG.set_bs_convergence_params(100, 1e-15, 1e-7);
    

    GMG.init();
    GMG.set_verbose(1, 0);
    GMG.set_num_prepostsmooth(3,3);
    GMG.set_cycle('V');
    
    GMG.solve(d_u, d_b, d_value);
    cudaDeviceSynchronize();

    cout << "Solver   ... DONE" << endl;



    /* ##################################################################
    #                           TDO                                     #
    ###################################################################*/


    TDO tdo(d_u, d_chi, h, dim, betastar, etastar, Assembly.getNumElements(), local_num_rows, d_A_local, d_node_index, Assembly.getGridSize(), rho, numLevels, p);
    tdo.init();
    tdo.set_verbose(0);
    tdo.innerloop(d_u, d_chi);    // get updated d_chi
    tdo.print_VTK(0);


    // vtk
    vector<double> chi(Assembly.getNumElements(), rho);
    vector<double> u(Assembly.getNumNodes() * dim, 0);
    string fileformat(".vtk");
    int file_index = 0;
    stringstream ss; 
    ss << "vtk/tdo";
    ss << file_index;
    ss << fileformat;

    if ( writeToVTK )
    {
        WriteVectorToVTK(chi, u, ss.str(), dim, Assembly.getNumNodesPerDim(), h, Assembly.getNumElements(), Assembly.getNumNodes() );
        
        CUDA_CALL( cudaMemcpy(&chi[0], d_chi, sizeof(double) * Assembly.getNumElements(), cudaMemcpyDeviceToHost) );
        CUDA_CALL( cudaMemcpy(&u[0], d_u, sizeof(double) * u.size(), cudaMemcpyDeviceToHost) );

        file_index++;
        ss.str( string() );
        ss.clear();
        ss << "vtk/tdo";
        ss << file_index;
        ss << fileformat;
        
        WriteVectorToVTK(chi, u, ss.str(), dim, Assembly.getNumNodesPerDim(), h, Assembly.getNumElements(), Assembly.getNumNodes() );
    }


    for ( int i = 1 ; i < 2 ; ++i )
    {
        // update the global stiffness matrix with the updated density distribution
        Assembly.UpdateGlobalStiffness(d_chi, d_value, d_index, d_p_value, d_p_index, d_r_value, d_r_index, d_A_local);

        cout << "Calculating iteration " << i << " ... " << endl;
        cudaDeviceSynchronize();
        GMG.reinit();
        GMG.set_convergence_params(10000, 1e-99, 1e-10);
        GMG.set_bs_convergence_params(1000, 1e-99, 1e-13);

        
        GMG.set_verbose(0, 0);
        GMG.solve(d_u, d_b, d_value);
        cudaDeviceSynchronize();

        // tdo.set_verbose(1);
        tdo.innerloop(d_u, d_chi);

        if ( writeToVTK )
        { 
            CUDA_CALL( cudaMemcpy(&chi[0], d_chi, sizeof(double) * Assembly.getNumElements(), cudaMemcpyDeviceToHost) );
            CUDA_CALL( cudaMemcpy(&u[0], d_u, sizeof(double) * u.size(), cudaMemcpyDeviceToHost) );

            file_index++;
            ss.str( string() );
            ss.clear();
            ss << "vtk/tdo";
            ss << file_index;
            ss << fileformat;
            
            WriteVectorToVTK(chi, u, ss.str(), dim, Assembly.getNumNodesPerDim(), h, Assembly.getNumElements(), Assembly.getNumNodes() );

        }

        cudaDeviceSynchronize();
    }

   
    cudaDeviceSynchronize();
    
}
