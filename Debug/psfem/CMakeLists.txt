cmake_minimum_required (VERSION 3.0)
project (PSFEM CUDA CXX)

## options
option(DEBUG "compile in debug mode" OFF)
option(PARALLEL "compile for parallel usage" OFF)
option(CUDA "compile for nvidia-GPU usage" ON)
message(STATUS "DEBUG       ${DEBUG}")
message(STATUS "PARALLEL    ${PARALLEL}")
message(STATUS "CUDA        ${CUDA}")

## directories
get_filename_component(ROOT_DIR ${CMAKE_CURRENT_LIST_FILE} PATH)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${ROOT_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${ROOT_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${ROOT_DIR}/lib)

if (CUDA)

	if(PARALLEL)
		message(FATAL_ERROR "PARALLEL option must be OFF.")

	else(PARALLEL)

		## CUDA
		find_package(CUDA)

		if(CUDA_FOUND)
		message("Compiling for CUDA implementation")
		
		set(CUDA_64_BIT_DEVICE_CODE ON)
		set(CMAKE_CUDA_FLAGS "--std=c++11 -O3")

		# Adding CUDA definition
		add_definitions(-DCUDA)

		set(SOURCES
			src/grid/structured_grid.cu
			src/grid/structured_multi_grid.cu
		)

		include_directories(src)
		add_library(psfemlib ${SOURCES})

		# add main to executable
		cuda_add_executable(psfem src/main.cu)

		set(linkLibraries)
		# add library to linking
		target_link_libraries(psfem psfemlib ${linkLibraries})

		else(CUDA_FOUND)
		message(FATAL_ERROR "CUDA not found.")

		endif(CUDA_FOUND)

	endif(PARALLEL)
		
else (CUDA)

	# cxx flags
	set(exclude_warnings "-Wno-unused-parameter -Wno-unused-variable")
	set(warnings "-Wall -Wextra")
	set(cxx_flags "${warnings} ${exclude_warnings} ${additionalFlags} -pedantic -std=c++11")

	if (DEBUG)
		set(cxx_flags "${cxx_flags} -g -D__DEBUG__ -O0")
	else (DEBUG)
		set(cxx_flags "${cxx_flags} -DNDEBUG -O3")
	endif (DEBUG)

	set(CMAKE_CXX_FLAGS "" CACHE STRING "clear flags" FORCE)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${cxx_flags}" CACHE STRING "overridden flags!" FORCE)


	set(linkLibraries)
	if (PARALLEL)
		add_definitions(-DPARALLEL)

		## MPI
		find_package(MPI)
		
		if (MPI_FOUND)
			include_directories(${MPI_INCLUDE_PATH})
			# Add mpi libraries:
			# Standard case: add cxx libraries
			if (MPI_CXX_LIBRARIES)
				set(linkLibraries ${linkLibraries} ${MPI_CXX_LIBRARIES})
			# Deprecated case: In order to support cmake versions < 2.8.6, 
			#                   where MPI_CXX_LIBRARIES cannot be used
			else (MPI_CXX_LIBRARIES)				
				if (MPI_LIBRARY)
					set(linkLibraries ${linkLibraries} ${MPI_LIBRARY})
				endif (MPI_LIBRARY)
				if (MPI_EXTRA_LIBRARY)
					set (linkLibraries ${linkLibraries} ${MPI_EXTRA_LIBRARY})
				endif (MPI_EXTRA_LIBRARY)
			endif (MPI_CXX_LIBRARIES)				
		else (MPI_FOUND)
			message(FATAL_ERROR "MPI not found.")
		endif (MPI_FOUND)
	endif (PARALLEL)


	# sources for the library
	set(SOURCES
		src/grid/structured_grid.cpp
		src/grid/structured_multi_grid.cpp
		src/solver/iterative_solver.cpp
		src/solver/cg.cpp
		src/precond/jacobi.cpp
		src/precond/gs.cpp
		src/precond/ilu.cpp
		src/precond/richardson.cpp
		src/precond/gmg/gmg.cpp
		src/precond/gmg/gmg_nested.cpp
		src/precond/gmg/prolongation.cpp
	)

	include_directories(src)
	add_library(psfemlib ${SOURCES})

	# add main to executable
	add_executable(psfem src/main.cpp)

	# add library to linking
	target_link_libraries(psfem psfemlib ${linkLibraries})

endif (CUDA)
