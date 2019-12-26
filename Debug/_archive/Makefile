
source = assemble_solve.cu cudakernels.cu
obj = assemble_solve.o cudakernels.o


# all: 
all: $(obj)
	nvcc $(obj) -o a

assemble_solve.o: assemble_solve.cu
	nvcc assemble_solve.cu -c

cudakernels.o: cudakernels.cu
	nvcc cudakernels.cu -c

clean:
	rm -f ${obj} *.o