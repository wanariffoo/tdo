/*
 * parallel.h
 *
 * author: a.vogel@rub.de
 */

#ifndef PARALLEL_H
#define PARALLEL_H

#include <memory>
#include <iostream>
#include <map>
#include <stdlib.h>							// Required for abort()
#include <chrono>


typedef int MPI_Comm;						
typedef int MPI_Op;
#define MPI_SUM     (MPI_Op)(0x58000003)
#define MPI_COMM_WORLD ((MPI_Comm)0x44000000)

namespace mpi{

class Comm{
	public:
		Comm(std::shared_ptr<MPI_Comm> spComm) 
			: m_spComm(spComm)
		{}

		int rank() const{return 0;}

		int size() const{return 1;}

		void abort(int errcode) const{
			std::cout << "Program aborted with error code: " << errcode << std::endl;
			std::abort();
		}

		void barrier() const{}

		double allreduce(double& d, MPI_Op op){

			return d;
		}


		void exchange(const std::map<int, std::vector<double>>& sendBufferMap,
					  std::map<int, std::vector<double>>& recvBufferMap,
					  int tag = 0)
		{
				if(!sendBufferMap.empty() || !recvBufferMap.empty()) {
					throw(std::runtime_error("Send/Receive Buffer Maps are not empty"));
					}		
					else { return; }
		}

	protected:
		std::shared_ptr<MPI_Comm> m_spComm;
};


inline bool initialized(){ return true;	}

inline bool finalized(){ return true; }

inline void init(int& argc, char**& argv){}	
	
inline void abort(int errcode){
	std::cout << "Program aborted with error code: " << errcode << std::endl;
	std::abort();									
}

inline void finalize(){}

inline double walltime(){

	std::chrono::seconds noTime{0};  										// TODO:
    std::chrono::system_clock::time_point systemEpoch{noTime};  

    auto t = std::chrono::high_resolution_clock::now();

	std::chrono::duration<double, std::milli> time = t - systemEpoch;

	return time.count() / 1000;
}

inline double walltick(){

	return 8.200000e-08;							
}													


inline Comm world(){
	return Comm(std::shared_ptr<MPI_Comm>(new MPI_Comm(MPI_COMM_WORLD)));
}


}

#endif // PARALLEL_H