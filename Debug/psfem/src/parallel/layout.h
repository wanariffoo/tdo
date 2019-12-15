/*
 * layout.h
 *
 * author: a.vogel@rub.de 
 *
 */

#ifndef LAYOUT_H
#define LAYOUT_H

#include <map>
#include <vector>

#include "parallel.h"

class Layout
{
	public:
		typedef std::vector<std::size_t> interface_type;
		typedef std::map<int, interface_type> map_type;
		typedef map_type::iterator iterator;
		typedef map_type::const_iterator const_iterator;

	public:
		const_iterator begin() const {return m_mapInterface.begin();}
		const_iterator end() const {return m_mapInterface.end();}

		iterator begin() {return m_mapInterface.begin();}
		iterator end() {return m_mapInterface.end();}

		void clear() {m_mapInterface.clear();}

		/// create new interface with indices or add to existing one
		void add_interface(int dest, const interface_type& vInd){
			std::size_t sz = vInd.size();
			if (!sz) return;

			interface_type& interface = m_mapInterface[dest];
			std::size_t prevSz = interface.size();
			interface.resize(prevSz + sz);
			for (std::size_t i = 0; i < sz; ++i)
				interface[prevSz + i] = vInd[i];
		}

	protected:
		map_type m_mapInterface;
};


class AlgebraLayout
{

	public:
		AlgebraLayout(mpi::Comm& comm) : m_comm(comm) {}

		const Layout& slave_layout() const {return m_slaveLayout;}
		Layout& slave_layout() {return m_slaveLayout;}

		const Layout& master_layout() const {return m_masterLayout;}
		Layout& master_layout() {return m_masterLayout;}


		mpi::Comm& comm() {return m_comm;}

		std::map<int, std::vector<double>>& slaveBufferMap() {return m_slaveBufferMap;}
		std::map<int, std::vector<double>>& masterBufferMap() {return m_masterBufferMap;}

	protected:
		mpi::Comm m_comm;

		Layout m_slaveLayout;
		Layout m_masterLayout;

		std::map<int, std::vector<double>> m_slaveBufferMap;
		std::map<int, std::vector<double>> m_masterBufferMap;
};


#endif // LAYOUT_H
