
# source = assemble_solve.cu cudakernels.cu solver.cu
obj = main.o cudakernels.o solver.o assemble.o tdo.o

# all: 
all: $(obj)
	nvcc $(obj) -o a

main.o: main.cu vtk.h
	nvcc main.cu -c

cudakernels.o: cudakernels.cu
	nvcc cudakernels.cu -c

solver.o: solver.cu
	nvcc solver.cu -c

assemble.o: assemble.cu
	nvcc assemble.cu -c

tdo.o: tdo.cu
	nvcc tdo.cu -c

clean:
	rm -f ${obj} *.o

delvtk:
	rm vtk/*.vtk

# assemble_solve.o: assemble_solve.cu
# 	nvcc assemble_solve.cu -c