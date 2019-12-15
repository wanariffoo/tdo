/*
 * parallel.h
 *
 * author: a.vogel@rub.de
 */

#ifndef PARALLEL_H
#define PARALLEL_H

#include <mpi.h>
#include <memory>
#include <map>

namespace mpi{


class Comm{
	public:
		Comm(std::shared_ptr<MPI_Comm> spComm) 
			: m_spComm(spComm)
		{}

		int rank() const{
			int rk;
			MPI_Comm_rank(*m_spComm, &rk);
			return rk;
		}

		int size() const{
			int sz;
			MPI_Comm_size(*m_spComm, &sz);
			return sz;
		}

		void abort(int errcode) const{
			MPI_Abort(*m_spComm, errcode);
		}

		void barrier() const{
			MPI_Barrier(*m_spComm);
		}

		double allreduce(double& d, MPI_Op op){
			if (size() <= 1) return d;

			double res;
			MPI_Allreduce(&d, &res, 1, MPI_DOUBLE, op, *m_spComm);
			return res;
		}


		void exchange(const std::map<int, std::vector<double>>& sendBufferMap,
					  std::map<int, std::vector<double>>& recvBufferMap,
					  int tag = 0)
		{
			if (size() <= 1) return;

			MPI_Comm comm = *m_spComm;

			// issue receives
			std::vector<MPI_Request> vRecvReq(recvBufferMap.size());
			int cnt = 0;
			for(auto it = recvBufferMap.begin(); it != recvBufferMap.end(); ++it, ++cnt)
			{
				const int fromRank = it->first;
				std::vector<double>& vData = it->second;
				if(vData.empty()) continue;
			
				MPI_Irecv(&vData[0], vData.size(), MPI_DOUBLE, fromRank, tag, comm, &vRecvReq[cnt]);
			}

			// issue sends
			cnt = 0;
			std::vector<MPI_Request> vSendReq(sendBufferMap.size());
			for(auto it = sendBufferMap.begin(); it != sendBufferMap.end(); ++it, ++cnt)
			{
				const int toRank = it->first;
				const std::vector<double>& vData = it->second;
				if(vData.empty()) continue;
			
				MPI_Isend(&vData[0], vData.size(), MPI_DOUBLE, toRank, tag, comm, &vSendReq[cnt]);
			}

			// wait until all operations have succeeded
			if (vRecvReq.size() > 0)
				MPI_Waitall(vRecvReq.size(), &vRecvReq[0], MPI_STATUSES_IGNORE);
			if (vSendReq.size() > 0)
				MPI_Waitall(vSendReq.size(), &vSendReq[0], MPI_STATUSES_IGNORE);
		}

	protected:
		std::shared_ptr<MPI_Comm> m_spComm;
};




inline bool initialized(){
	int flag;
  	MPI_Initialized(&flag);
	return flag != 0;
}

inline bool finalized(){
	int flag;
	MPI_Finalized(&flag);
	return flag != 0;
}

inline void init(int& argc, char**& argv){
	if(!initialized())
		MPI_Init(&argc, &argv);
}

inline void abort(int errcode){
	MPI_Abort(MPI_COMM_WORLD, errcode);
}

inline void finalize(){
	if(initialized() && !finalized())
		MPI_Finalize();
}

inline double walltime(){
	return MPI_Wtime();
}

inline double walltick(){
	return MPI_Wtick();
}


inline Comm world(){
	return Comm(std::shared_ptr<MPI_Comm>(new MPI_Comm(MPI_COMM_WORLD)));
}


}

#endif // PARALLEL_H