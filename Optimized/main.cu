/*
    main.cu

    Developed for the master thesis project: GPU-accelerated Thermodynamic Topology Optimization
    Author: Wan Arif bin Wan Abhar
    Institution: Ruhr Universitaet Bochum
*/

#include <iostream>
#include <vector>
#include "include/cudakernels.h"
#include "include/assemble.h"
#include "include/solver.h"
#include "include/tdo.h"
#include "include/vtk.h"
#include <iomanip>
#include <ctime>

using namespace std;

int main()
{

    // create vtk files
    bool writeToVTK = true;

    // include in output file: compliance, stiffness and MOD
    bool CSMOD = true;

    // material properties
    double youngMod = 200000;
    double poisson = 0.33;

    //// model set-up
    size_t numLevels = 6;
    
    vector<size_t> N;
    vector<vector<size_t>> bc_index(numLevels);

    // number of steps
    size_t update_steps = 100;
    double c_tol = 1e-8;

    // prints result on terminal
    bool gmg_verbose = 0; // first iteration of precond
    bool bs_verbose = 0; // first iteration of base-solver
    bool gmg_verbose_ = 0; // subsequent optimization steps
    bool bs_verbose_ = 0; // subsequent optimization steps


    // CASE 0 : 2D MBB
    N = {3,1};                  // domain dimension (x,y,z) on coarsest grid
    double h_coarse = 1;        // local element mesh size on coarsest grid
    size_t bc_case = 0;
    double damp = 2.0/3.0;      // smoother (jacobi damping parameter)


    // // CASE 1 : 3D MBB
    // N = {6,2,1};                // domain dimension (x,y,z) on coarsest grid
    // double h_coarse = 0.5;      // local element mesh size on coarsest grid
    // size_t bc_case = 1;
    // double damp = 1.0/2.0;      // smoother (jacobi damping parameter)

    
    // applying boundary conditions
    size_t dim = N.size();
    bc_index = applyBC(N, numLevels, bc_case, dim);

    // calculating the mesh size on the top level grid
    double h = h_coarse/pow(2,numLevels - 1);
    size_t local_num_rows = pow(2,dim)*dim;

    // density update variables
    double rho = 0.3;
    size_t p = 3;
    double etastar = 12.0;
    double betastar = 2 * pow(h,2);

    // values for the global stiffnes matrix, A
    vector<size_t> num_rows;
    vector<size_t> max_row_size;

    vector<size_t> p_max_row_size;

    //// device pointers
    // local stiffness
    double* d_A_local;

    // global stiffness matrix on each grid-level
    vector<double*> d_value;
    vector<size_t*> d_index;

    // prolongation matrices
    vector<double*> d_p_value;
    vector<size_t*> d_p_index;

    // design variable
    double* d_chi;

    vector<size_t*> d_node_index;
    size_t* d_node_index_;

    //// benchmarking stuff
    // cuda event
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds;

    // creating output file
    // obtaining time and date for output filename
    time_t now = time(0);
    tm *ltm = localtime(&now);
    
    string fileformat_(".txt");
    stringstream ssbm; 
    ssbm << "outputs/" << 1900 + ltm->tm_year << "_" << 1 + ltm->tm_mon << "_" << ltm->tm_mday << "_" << 2 + ltm->tm_hour << ltm->tm_min << 1 + ltm->tm_sec << "_" << dim << "d_lvl_" << numLevels;
    ssbm << fileformat_;
    ofstream ofssbm(ssbm.str(), ios::out);


    // output file
    ofssbm << "### GPU-accelerated Thermodynamic Topology Optimization ###" << endl;
    ofssbm << "Dimension: " << dim << endl;
    ofssbm << "Number of Multigrid Levels: " << numLevels << endl;


    // for overall benchmark
    cudaEvent_t start_, stop_;
    cudaEventCreate(&start_);
    cudaEventCreate(&stop_);
    float milliseconds_;
    cudaEventRecord(start_);

    /* ##################################################################
    #                           ASSEMBLY                                #
    ###################################################################*/
       

    Assembler Assembly(dim, h, N, youngMod, poisson, rho, p, numLevels);
    Assembly.setBC(bc_index);

        cudaEventRecord(start);
    Assembly.init_GPU(d_A_local, d_value, d_index, d_p_value, d_p_index, d_chi, num_rows, max_row_size, p_max_row_size, d_node_index, d_node_index_, ofssbm);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        ofssbm << endl;
        ofssbm << "Total assembly time\t\t" << milliseconds << endl;



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

    Solver GMG(d_value, d_index, max_row_size, d_p_value, d_p_index, p_max_row_size, numLevels, num_rows, damp);
    
    
    GMG.set_convergence_params(100000, 1e-15, 1e-8);
    GMG.set_bs_convergence_params(1000, 1e-15, 1e-7);
    GMG.init();
    GMG.set_verbose(gmg_verbose,bs_verbose);
    GMG.set_num_prepostsmooth(3,3);
    GMG.set_cycle('V');
            
                ofssbm << endl;
                ofssbm << "SOLVER" << endl;
                cudaEventRecord(start);
    GMG.solve(d_u, d_b, d_value, ofssbm);
                cudaEventRecord(stop);
                cudaEventSynchronize(stop);
                milliseconds = 0;
                cudaEventElapsedTime(&milliseconds, start, stop);
                ofssbm << endl;
                ofssbm << "Total solver time\t\t" << milliseconds << endl;
            

    cout << "Solver   ... DONE" << endl;

    /* ##################################################################
    #                         DENSITY UPDATE                            #
    ###################################################################*/

    // structure compliance
    double c = 0;
    double last_c = 0;
    double MOD;
    double* d_c;
    double* d_MOD;
    CUDA_CALL( cudaMalloc((void**)&d_c, sizeof(double) ) );
    CUDA_CALL( cudaMalloc((void**)&d_MOD, sizeof(double) ) );


    TDO tdo(d_u, d_chi, h, dim, betastar, etastar, Assembly.getNumElements(), local_num_rows, d_A_local, d_node_index, Assembly.getGridSize(), rho, numLevels, p, d_node_index_);
    tdo.init();
    tdo.set_verbose(0);
                ofssbm << endl;
                ofssbm << "DENSITY UPDATE" << endl;
                cudaEventRecord(start);
    tdo.innerloop(d_u, d_chi, d_c, d_MOD, ofssbm);    // get updated d_chi
                cudaEventRecord(stop);
                cudaEventSynchronize(stop);
                milliseconds = 0;
                cudaEventElapsedTime(&milliseconds, start, stop);
                ofssbm << endl;
                ofssbm << "Total density update time\t" << milliseconds << endl;
    tdo.print_VTK(0);
    last_c = c;
    CUDA_CALL( cudaMemcpy(&c, d_c, sizeof(double), cudaMemcpyDeviceToHost) );
    CUDA_CALL( cudaMemcpy(&MOD, d_MOD, sizeof(double), cudaMemcpyDeviceToHost) );

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
    
    
    size_t iterations = 1;
    float sum_a = 0;  // sum of assembly time
    float sum_s = 0;  // sum of solver time
    float sum_du = 0; // sum of density update time
    float c_rel = abs( c - last_c) / c;
    float init_E = c * pow(rho,p) * youngMod;
    
    ofssbm << endl;
    ofssbm << "assembly_total,solver_total,solver_average,solver_no_iter,d_update_total,total_bisection,bisection_no_iter,avg_bisection" << endl;


    cudaEventRecord(stop_);
    cudaEventSynchronize(stop_);
    milliseconds_ = 0;
    cudaEventElapsedTime(&milliseconds_, start_, stop_);
    float elapsed_time = milliseconds_;


    
    for ( int i = 1 ; i < (update_steps) ; ++i )
    // for ( int i = 1 ; i < (update_steps) && ( c_rel > c_tol ); ++i )
    {
        cudaEventRecord(start_);
        ofssbm << iterations << ",";

        // update the global stiffness matrix with the updated density distribution
        cudaEventRecord(start);
        Assembly.UpdateGlobalStiffness(d_chi, d_value, d_index, d_p_value, d_p_index, d_A_local);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        ofssbm << milliseconds << ",";
        sum_a += milliseconds;


        cout << "Calculating iteration " << i << " ... " << endl;
        // cudaDeviceSynchronize();
        cudaEventRecord(start);
        GMG.reinit();
        GMG.setBM(true);
        GMG.set_convergence_params(1500000, 1e-99, 1e-10);
        GMG.set_bs_convergence_params(1000, 1e-99, 1e-13);
        GMG.set_verbose(gmg_verbose_, bs_verbose_);
        GMG.solve(d_u, d_b, d_value, ofssbm);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        ofssbm << milliseconds << "," << milliseconds/GMG.getCounter() << "," << GMG.getCounter() << ",";
        sum_s += milliseconds;

        // cout << "Solver done ... " << endl;
        cudaEventRecord(start);
        tdo.setBM(true);
        tdo.innerloop(d_u, d_chi, d_c, d_MOD, ofssbm);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        ofssbm << milliseconds << "," << tdo.getSum() << "," << tdo.getCounter() << "," << tdo.getSum()/tdo.getCounter();
        sum_du += milliseconds;
        
        cudaEventRecord(stop_);
        cudaEventSynchronize(stop_);
        milliseconds_ = 0;
        cudaEventElapsedTime(&milliseconds_, start_, stop_);
        elapsed_time += milliseconds_;

        if (!CSMOD)
        ofssbm << "," << elapsed_time << endl;

        // collect data of compliance, stiffness and MOD
        else
        {
            // convergence check with compliance
            last_c = c;
            CUDA_CALL( cudaMemcpy(&c, d_c, sizeof(double), cudaMemcpyDeviceToHost) );
            c_rel = abs( c - last_c ) / c;
            ofssbm << "," << c;

            // structural stiffness, E = 1 / c * E_init
            ofssbm << "," << init_E / c;
            
            // computing MOD
            CUDA_CALL( cudaMemcpy(&MOD, d_MOD, sizeof(double), cudaMemcpyDeviceToHost) );
            ofssbm << "," << MOD;

            ofssbm << "," << elapsed_time << endl;
        }

        // cout << "Density update done ... " << endl;

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



        iterations++;
        
    }


    ofssbm << endl;
    ofssbm << "Average assembly time\t\t" << sum_a / (iterations - 1) << endl;
    ofssbm << "Average solver time\t\t" << sum_s / (iterations - 1) << endl;
    ofssbm << "Average density update time\t" << sum_du / (iterations - 1)  << endl;
    ofssbm << endl;
    ofssbm << "Number of TDO iterations\t" << (iterations - 1) << endl;
    ofssbm << "Time per TDO step \t\t" << elapsed_time/(iterations - 1) << endl;
    ofssbm << "TOTAL RUNTIME\t\t\t" << elapsed_time << endl;
   
    
    
    // memory deallocation

    CUDA_CALL( cudaFree(d_A_local) );  
    CUDA_CALL( cudaFree(d_chi) );
    // CUDA_CALL( cudaFree(d_node_index) );
    CUDA_CALL( cudaFree(d_u) );
    CUDA_CALL( cudaFree(d_b) );
    CUDA_CALL( cudaFree(d_MOD) );
    CUDA_CALL( cudaFree(d_c) );

    for ( int i = 0 ; i < numLevels ; i++)
    {
        CUDA_CALL( cudaFree(d_value[i]) );
        CUDA_CALL( cudaFree(d_index[i]) );
    }

    for ( int i = 0 ; i < numLevels -1 ; i++)
    {
        CUDA_CALL( cudaFree(d_p_value[i]) );
        CUDA_CALL( cudaFree(d_p_index[i]) );
    }



    
    
}
