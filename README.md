# GPU-accelerated thermodynamic topology optimization

Developed for a master thesis project by Wan Arif bin Wan Abhar (wan.wanabhar@rub.de) at Ruhr Universität Bochum in SoSe-2019/20 term.


## Table of contents
* [Introduction](https://github.com/wanariffoo/tdo#introduction)
* [Inputting the parameters](https://github.com/wanariffoo/tdo#inputting-the-parameters)
* [Compiling and running the program](https://github.com/wanariffoo/tdo#compiling-and-running-the-program)
* [Output file](https://github.com/wanariffoo/tdo#output-file)

## Introduction

The program solves a topology optimization problem based on the thermodynamic topology optimization approach by [Jantos, Hackl & Junker (2018)](https://onlinelibrary.wiley.com/doi/full/10.1002/nme.5988) using the GPU.

The program solves two cases: a 2-dimensional MBB beam and a 3-dimensional cantilever, with the dimensions and boundary conditions illustrated below:

* 2-dimensional MBB beam

<img src="https://github.com/wanariffoo/tdo/blob/master/Optimized/vtk/2d_case.png" alt="2D MBB" width="280">

* 3-dimensional cantilever

<img src="https://github.com/wanariffoo/tdo/blob/master/Optimized/vtk/3d_case.png" alt="3D cantilever" width="250">


The program offers as well an option for a visualization from `vtk` files. The *Paraview* software is used to visualize the optimization process:

* 2-dimensional MBB beam

<img src="https://i.makeagif.com/media/6-27-2020/5ZnEre.gif" alt="2D MBB" width="380"> 

* 3-dimensional cantilever

<img src="https://i.makeagif.com/media/6-27-2020/OXDitS.gif" alt="3D cantilever" width="300">

## Inputting the parameters
The parameters are inputted in the `Optimized/main.cu` file:
- `writeToVTK` : set to `true` to create the .vtk visualization files, which are saved in the `vtk` folder
- `CSMOD` : set to `true` to output the compliance, stiffness and the MOD in each iteration step
- `youngMod` : Young's Modulus
- `poisson` : Poisson ratio
- `numLevels` : number of grid levels used in the GMG preconditioner
- `update_steps` : maximum number of iteration steps
- `c_tol` : value of the compliance tolerance
- `N` : the grid dimension of the domain. Note that only {3,1} and {6,2,1} are available for now
- `rho` : prescribed relative structure volume
- `p` : penalization exponent
- `beta_star` : regularization parameter
- `eta_star` : viscosity

## Compiling and running the program
To compile and run, type `make` in the `Optimized` folder and run the executable file `tdo`.

```
$ cd Optimized
$ make
$ ./tdo
```

In the project, the program was run on an Nvidia V100 PCIe graphics card, with a *Compute Compatibility* of 7.0. If a newer GPU is used, change accordingly the *Compute Compatibility* `-arch=sm_XX` on line 5 in the `Makefile` file.

## Output file
An output file is produced in the `outputs` folder each time the program is executed. The output file contains the time measurements of the processes involved during the optimization process. The time taken for each kernel is taken in the first iteration step, as seen below. For subsequent iteration steps, only the time taken of each process is displayed. The data can be copied and pasted into a spreadsheet software separated by comma, in order to have a clean presentation of the data.

```
### GPU-accelerated Thermodynamic Topology Optimization ###
Dimension: 2
Number of Multigrid Levels: 11
Top-level grid size = { 3072, 1024 }
Top-level number of rows = 6299650
Number of Elements = 3149825

(All measurements in ms)

ASSEMBLER
assembleLocal() 		0.053472
assembleProlMatrix_GPU() 	27.7283
fillIndexVector() 		2.4504
assembleGlobalStiffness_GPU() 	48.1283
applyMatrixBC_GPU() 		0.136256
PTAP() 				105.716
Memory allocation & copy	4821.13

Total assembly time		5000.23

SOLVER
ComputeResiduum_GPU()		2.71264
norm_GPU()			14.5832
Smoother : Jacobi()		0.982656
ApplyTransposed_GPU()		25.732
   Base: ComputeResiduum_GPU()	0.010432
   Base: norm_GPU()		0.011424
   Base: Jacobi_Precond_GPU()	0.009536
   Base: dotProduct()		0.035392
   Base: calcDirectionVector()	0.018976
   Base: Apply_GPU()		0.015808
   Base: calculateAlpha()	0.036608
   Base: axpy_GPU()		0.013184
   Base: axpy_neg_GPU()		0.015904
Apply_GPU()			0.63264
UpdateResiduum_GPU()		2.70912
dotProduct()		        0.154112
calcDirectionVector()	        0.135968
Apply_GPU()		        2.7473
calculateAlpha()	        0.169056
axpy_GPU()		        0.191904
axpy_neg_GPU()		        0.194784

Total solver time		1227.88

DENSITY UPDATE
calcDrivingForce()		1.54704
calcP_w()			0.151104
calcEtaBeta()			0.01456
calcLambdaLower()		0.188672
calcLambdaLower()		0.194976
calcChiTrial()			0.115776
calcRhoTrial()			0.074592
calcLambdaTrial()		0.014336
Total time of bisection algo 	7.77261
Number of steps 		31
Average time per bisection step 0.250729

Total density update time	12.06

assembly_total,solver_total,solver_average,solver_no_iter,d_update_total,total_bisection,bisection_no_iter,avg_bisection
1,155.809,2069.1,94.0499,22,10.0919,5.91846,24,0.246603,0.00557261,12513.6,0.80007,59166.8
2,155.702,2075.01,94.3189,22,9.72922,5.59062,23,0.243071,0.00394439,17679.2,0.763575,61407.3
3,155.763,2432.45,93.5558,26,9.75456,5.62547,23,0.244586,0.00327144,21315.8,0.736794,64005.3

...

```
