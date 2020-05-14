

# NVCC flags
NVCC 		= nvcc
NVCC_FLAGS 	= -arch=sm_70 

# Directories
SRC_DIR = src
INC_DIR = include
OBJ_DIR = bin
VTK_DIR = vtk

# Object files
OBJS = $(OBJ_DIR)/main.o $(OBJ_DIR)/cudakernels.o $(OBJ_DIR)/solver.o $(OBJ_DIR)/assemble.o $(OBJ_DIR)/tdo.o

# all: 
all : $(OBJS)
	nvcc $(OBJS) -o tdo

$(OBJ_DIR)/main.o : main.cu $(INC_DIR)/vtk.h
	nvcc $(NVCC_FLAGS) main.cu -c -o $@

$(OBJ_DIR)/cudakernels.o: $(SRC_DIR)/cudakernels.cu
	nvcc $(NVCC_FLAGS) $(SRC_DIR)/cudakernels.cu -c -o $@

$(OBJ_DIR)/solver.o: $(SRC_DIR)/solver.cu
	nvcc $(NVCC_FLAGS) $(SRC_DIR)/solver.cu -c -o $@

$(OBJ_DIR)/assemble.o: $(SRC_DIR)/assemble.cu
	nvcc $(NVCC_FLAGS) $(SRC_DIR)/assemble.cu -c -o $@

$(OBJ_DIR)/tdo.o: $(SRC_DIR)/tdo.cu
	nvcc $(NVCC_FLAGS) $(SRC_DIR)/tdo.cu -c -o $@

# deletes only the vtk files
vtk:
	rm -f $(VTK_DIR)/*.vtk

# clears all object and vtk files
clean:
	rm -f ${OBJS} $(OBJ_DIR)/*.o $(VTK_DIR)/*.vtk


