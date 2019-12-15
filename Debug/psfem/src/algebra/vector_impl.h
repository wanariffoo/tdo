/*
 * vector_impl.h
 *
 * author: 	markus.breit@gcsc.uni-frankfurt.de
 * 			a.vogel@rub.de
 *
 */

#ifndef VECTOR_IMPL_H
#define VECTOR_IMPL_H


#include "algebra/vector.h"
#include "parallel/parallel.h"
#include <cassert>
#include <cmath>
#include <ctime>
#include <iostream>

template <class T>
Vector<T>::Vector(std::shared_ptr<AlgebraLayout> spLayouts) 
 : m_storageMask(PST_UNDEFINED), m_spAlgebraLayout(spLayouts)
{}

template <class T>
Vector<T>::Vector(std::size_t size, double val, std::shared_ptr<AlgebraLayout> spLayouts) 
: m_storageMask(PST_CONSISTENT), m_spAlgebraLayout(spLayouts)
{
	resize(size, val);
	if(val == 0.0)
		m_storageMask = PST_CONSISTENT | PST_ADDITIVE | PST_UNIQUE;
}


template <class T>
Vector<T>& Vector<T>::operator=(double d)
{
	const std::size_t sz = size();
	for (std::size_t i = 0; i < sz; ++i)
		operator[](i) = d;

	if(d == 0.0)
		m_storageMask = PST_CONSISTENT | PST_ADDITIVE | PST_UNIQUE;
	else
		m_storageMask = PST_CONSISTENT;

	return *this;
}


template <class T>
Vector<T>& Vector<T>::operator+=(const Vector& v)
{
	check_size_matches(v);

	const std::size_t sz = size();
	for (std::size_t i = 0; i < sz; ++i)
		operator[](i) += v[i];

	// compute combined storage mask
	m_storageMask &= v.get_storage_mask();

	return *this;
}

template <class T>
Vector<T>& Vector<T>::operator-=(const Vector& v)
{
	check_size_matches(v);

	const std::size_t sz = size();
	for (std::size_t i = 0; i < sz; ++i)
		operator[](i) -= v[i];

	// compute combined storage mask
	m_storageMask &= v.get_storage_mask();

	return *this;
}

template <class T>
Vector<T>& Vector<T>::axpy(double s, const Vector& v)
{
	check_size_matches(v);

	const std::size_t sz = size();
	for (std::size_t i = 0; i < sz; ++i)
		operator[](i) += s * v[i];

	// compute combined storage mask
	m_storageMask &= v.get_storage_mask();

	return *this;
}


template <class T>
Vector<T>& Vector<T>::operator*=(double d)
{
	const std::size_t sz = size();
	for (std::size_t i = 0; i < sz; ++i)
		operator[](i) *= d;

	return *this;
}


template <class T>
double Vector<T>::operator*(const Vector& v) const
{
	check_size_matches(v);

	// compute and check combined storage mask
	if (m_spAlgebraLayout->comm().size() > 1)
	{
		// the result can only be calculated if the storage type
		// (1) of both vectors is unique or
		// (2) of one vector is consistent and the other is additive (unique)
		assert(((has_storage_type(PST_CONSISTENT) && v.has_storage_type(PST_ADDITIVE))
			|| (v.has_storage_type(PST_CONSISTENT) && has_storage_type(PST_ADDITIVE))
			|| (has_storage_type(PST_UNIQUE) && v.has_storage_type(PST_UNIQUE)))
			&& "Either bot vectors must be PST_UNIQUE or one CONSISTENT and the other PST_ADDITIVE.");
	}

	// process-local products
	const std::size_t sz = size();
	double res = 0.0;
	for (std::size_t i = 0; i < sz; ++i)
		res += operator[](i) * v[i];

	if (m_spAlgebraLayout->comm().size() > 1)
		return m_spAlgebraLayout->comm().allreduce(res, MPI_SUM);

	return res;
}

template <class T>
double Vector<T>::norm() const
{
	// make unique
	change_storage_type(PST_UNIQUE);

	// compute process-local dot products
	const std::size_t sz = size();
	double res = 0.0;
	for (std::size_t i = 0; i < sz; ++i)
		res += operator[](i) * operator[](i);

	// sum up local results
	res = m_spAlgebraLayout->comm().allreduce(res, MPI_SUM);
	return sqrt(res);
} 


template <class T>
void Vector<T>::random(T min, T max) 
{
	std::srand(std::time(0));
	const std::size_t sz = size();
	for (std::size_t i = 0; i < sz; ++i)
		operator[](i) = min + (std::rand() / (double) RAND_MAX) * (max - min);

	set_storage_type(PST_CONSISTENT);
	change_storage_type(PST_UNIQUE);
	change_storage_type(PST_CONSISTENT);
} 


template <class T>
void Vector<T>::check_size_matches(const Vector& v) const
{
	assert(v.size() == size() && "Vector size mismatch.");
}


template <class T>
void Vector<T>::set_storage_type(uint8_t type)
{
	if(type == PST_UNIQUE) type = PST_UNIQUE | PST_ADDITIVE;
	m_storageMask = type;
}


template <class T>
bool Vector<T>::has_storage_type(uint8_t type) const
{
	return type == PST_UNDEFINED ? m_storageMask == PST_UNDEFINED : (m_storageMask & type) == type;
}

template <class T>
uint8_t Vector<T>::get_storage_mask() const
{
	return m_storageMask;
}


template <class T>
bool Vector<T>::change_storage_type(ParallelStorageType newType) const
{
	// note: we allow to change the storage type of a const vector, since it 
	//       will not change the globally represented values 
	//       There reason that we do not use mutable data members for this 
	//       purpose but a const-cast is, that several non-const methods are 
	//       required to implement this method
	return const_cast<Vector<T>*>(this)->change_storage_type(newType);
}

template <class T>
bool Vector<T>::change_storage_type(ParallelStorageType newType)
{
	// if the current state is unknown, we cannot do anything
	if (m_storageMask == PST_UNDEFINED)
		return false;

	// if the current state already contains the required one, we need not do anything
	if (has_storage_type(newType))
		return true;

	// now change the storage type (if possible)
	if (newType == PST_CONSISTENT)
	{
		// if we are unique, we only have to copy from masters to slaves
		if (has_storage_type(PST_UNIQUE))
		{
			copy_master_to_slaves();
			set_storage_type(PST_CONSISTENT);
			return true;
		}

		// if we are additive, we need to add from slaves to master and then copy back
		if (has_storage_type(PST_ADDITIVE))
		{
			add_slaves_to_master();
			copy_master_to_slaves();
			set_storage_type(PST_CONSISTENT);
			return true;
		}

		// in case anything else is the case: no success
		return false;
	}

	if (newType == PST_ADDITIVE)
	{
		// we always make the vector unique, because that is additive a fortiori;
		// if our vector already is unqiue, we only add the additive flag
		if (has_storage_type(PST_UNIQUE))
		{
			set_storage_type(PST_ADDITIVE | PST_UNIQUE);
			return true;
		}

		// if we are consistent, set slaves to zero
		if (has_storage_type(PST_CONSISTENT))
		{
			set_slaves_zero();
			set_storage_type(PST_ADDITIVE | PST_UNIQUE);
			return true;
		}

		// in case anything else is the case: no success
		return false;
	}

	if (newType == PST_UNIQUE)
	{
		// if we are additive, then add slaves to master and then set slaves to zero
		if (has_storage_type(PST_ADDITIVE))
		{
			add_slaves_to_master();
			set_slaves_zero();
			set_storage_type(PST_ADDITIVE | PST_UNIQUE);
			return true;
		}

		// if we are consistent, set slaves to zero
		if (has_storage_type(PST_CONSISTENT))
		{
			set_slaves_zero();
			set_storage_type(PST_ADDITIVE | PST_UNIQUE);
			return true;
		}

		// in case anything else is the case: no success
		return false;
	}

	// in case anything else is the case: no success
	return false;
}

template <class T>
void Vector<T>::add_slaves_to_master()
{
//	std::cout << "CALLED: add_slaves_to_master" << std::endl;

	if(m_spAlgebraLayout == nullptr)
		throw(std::runtime_error("No algebraic layouts present"));

	// only do anything if we are parallel
	if (m_spAlgebraLayout->comm().size() <= 1) return;

	const Layout& masterLayout = m_spAlgebraLayout->master_layout();
	const Layout& slaveLayout = m_spAlgebraLayout->slave_layout();

	// get send buffer and clear them (important: clear does not free memory)
	std::map<int, std::vector<double>>& sendBufferMap = m_spAlgebraLayout->slaveBufferMap();
//	for(auto it = sendBufferMap.begin(); it != sendBufferMap.end(); ++it)
//		it->second.clear();

	// resize send buffer and copy values into buffer
	for (auto it = slaveLayout.begin(); it != slaveLayout.end(); ++it)
	{
		const int toRank = it->first;
		const std::vector<std::size_t>& interface = it->second;
		std::vector<double>& sendBuffer = sendBufferMap[toRank];

		if(sendBuffer.size() != interface.size()){
//			std::cout << "add_slaves_to_master -- sendBuffer RESIZE" << std::endl;
			sendBuffer.resize(interface.size());
		}

		for (std::size_t i = 0; i < interface.size(); ++i)
			sendBuffer[i] = (*this)[ interface[i] ];
	}

	// get recv buffer and clear them (important: clear does not free memory)
	std::map<int, std::vector<double>>& recvBufferMap = m_spAlgebraLayout->masterBufferMap();
//	for(auto it = recvBufferMap.begin(); it != recvBufferMap.end(); ++it)
//		it->second.clear();

	// resize recv buffer
	for (auto it = masterLayout.begin(); it != masterLayout.end(); ++it)
	{
		const int fromRank = it->first;
		const std::vector<std::size_t>& interface = it->second;
		std::vector<double>& recvBuffer = recvBufferMap[fromRank];

		if(recvBuffer.size() != interface.size()){
//			std::cout << "add_slaves_to_master -- recvBuffer RESIZE" << std::endl;
			recvBuffer.resize(interface.size());
		}
	}

	// exchange data
	m_spAlgebraLayout->comm().exchange(sendBufferMap, recvBufferMap);

	// add recv values into this vector
	for (auto it = masterLayout.begin(); it != masterLayout.end(); ++it)
	{
		const int fromRank = it->first;
		const std::vector<std::size_t>& interface = it->second;
		std::vector<double>& recvBuffer = recvBufferMap[fromRank];

		std::size_t sz = interface.size();
		for (std::size_t i = 0; i < sz; ++i)
			(*this)[ interface[i] ] += recvBuffer[i];
	}
}

template <class T>
void Vector<T>::copy_master_to_slaves()
{
//	std::cout << "CALLED: copy_master_to_slaves" << std::endl;

	if(m_spAlgebraLayout == nullptr)
		throw(std::runtime_error("No algebraic layouts present"));

	// only do anything if we are parallel
	if (m_spAlgebraLayout->comm().size() <= 1) return;

	const Layout& masterLayout = m_spAlgebraLayout->master_layout();
	const Layout& slaveLayout = m_spAlgebraLayout->slave_layout();

	// get send buffer and clear them (important: clear does not free memory)
	std::map<int, std::vector<double>>& sendBufferMap = m_spAlgebraLayout->masterBufferMap();
//	for(auto it = sendBufferMap.begin(); it != sendBufferMap.end(); ++it)
//		it->second.clear();

	// resize send buffer and copy values into buffer
	for (auto it = masterLayout.begin(); it != masterLayout.end(); ++it)
	{
		const int toRank = it->first;
		const std::vector<std::size_t>& interface = it->second;
		std::vector<double>& sendBuffer = sendBufferMap[toRank];

		if(sendBuffer.size() != interface.size()){
//			std::cout << "copy_master_to_slaves -- sendBuffer RESIZE" << std::endl;
			sendBuffer.resize(interface.size());
		}

		for (std::size_t i = 0; i < interface.size(); ++i)
			sendBuffer[i] = (*this)[ interface[i] ];
	}

	// get recv buffer and clear them (important: clear does not free memory)
	std::map<int, std::vector<double>>& recvBufferMap = m_spAlgebraLayout->slaveBufferMap();
//	for(auto it = recvBufferMap.begin(); it != recvBufferMap.end(); ++it)
//		it->second.clear();

	// resize recv buffer
	for (auto it = slaveLayout.begin(); it != slaveLayout.end(); ++it)
	{
		const int fromRank = it->first;
		const std::vector<std::size_t>& interface = it->second;
		std::vector<double>& recvBuffer = recvBufferMap[fromRank];

		if(recvBuffer.size() != interface.size()){
//			std::cout << "copy_master_to_slaves -- recvBuffer RESIZE" << std::endl;
			recvBuffer.resize(interface.size());
		}
	}

	// exchange data
	m_spAlgebraLayout->comm().exchange(sendBufferMap, recvBufferMap);

	// add recv values into this vector
	for (auto it = slaveLayout.begin(); it != slaveLayout.end(); ++it)
	{
		const int fromRank = it->first;
		const std::vector<std::size_t>& interface = it->second;
		std::vector<double>& recvBuffer = recvBufferMap[fromRank];

		std::size_t sz = interface.size();
		for (std::size_t i = 0; i < sz; ++i)
			(*this)[ interface[i] ] = recvBuffer[i];
	}
}

template <class T>
void Vector<T>::set_slaves_zero()
{
//	std::cout << "CALLED: set_slaves_zero" << std::endl;

	if(m_spAlgebraLayout == nullptr)
		throw(std::runtime_error("No algebraic layouts present"));

	// only do anything if we are parallel
	if (m_spAlgebraLayout->comm().size() <= 1) return;

	// get layout
	const Layout& slaveLayout = m_spAlgebraLayout->slave_layout();

	// loop layouts
	auto itEnd = slaveLayout.end();
	for (auto it = slaveLayout.begin(); it != itEnd; ++it)
	{
		// get slave interface
		const std::vector<std::size_t>& interface = it->second;

		// set vector entry to zero for all interface indices
		for (std::size_t i = 0; i < interface.size(); ++i)
			(*this)[  interface[i] ] = 0.0;
	}
}


// output for vector
template <typename T>
std::ostream& operator<<(std::ostream& stream, const Vector<T>& v)
{
	if (!v.size()) return stream << "()";

	std::size_t sz = v.size() - 1;
	stream << "(";
	for (std::size_t i = 0; i < sz; ++i)
		stream << v[i] << " ";
	stream << v[sz] << ")";

	return stream;
}


#endif // VECTOR_IMPL_H

