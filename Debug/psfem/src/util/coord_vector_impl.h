/*
 * coord_vector.cpp
 *
 * author: a.vogel@rub.de
 *
 */
 
#ifndef COORD_VECTOR_IMPL_H
#define COORD_VECTOR_IMPL_H

#include "coord_vector.h"

#include <cassert>	// for assert
#include <cmath>	// for sqrt

template <std::size_t dim, typename T>
CoordVector<dim, T>::CoordVector()
{
	for (std::size_t i = 0; i < dim; ++i)
		m_v[i] = 0.0;
	std::vector<int> v;
}  

template <std::size_t dim, typename T>
CoordVector<dim, T>::CoordVector(const CoordVector<dim, T>& v)
{
	for (std::size_t i = 0; i < dim; ++i)
		m_v[i] = v.m_v[i];
}

template <std::size_t dim, typename T>
CoordVector<dim, T>& CoordVector<dim, T>::operator=(const CoordVector<dim, T>& v)
{
    for (std::size_t i = 0; i < dim; ++i)
        m_v[i] = v.m_v[i];
    return *this;
}

template <std::size_t dim, typename T>
CoordVector<dim, T> CoordVector<dim, T>::operator*(const T& s) const
{
	CoordVector<dim, T> res;
    for (std::size_t i = 0; i < dim; ++i)
        res[i] = m_v[i] * s;
    return res;
}

template <std::size_t dim, typename T>
CoordVector<dim, T>::CoordVector(T d)
{
	for (std::size_t i = 0; i < dim; ++i)
		m_v[i] = d;
}

template <std::size_t dim, typename T>
const T& CoordVector<dim, T>::operator[](std::size_t i) const
{
	assert(i < dim && "Tried to access invalid coordinate vector index.");
	return m_v[i];
}

template <std::size_t dim, typename T>
T& CoordVector<dim, T>::operator[](std::size_t i)
{
	assert(i < dim && "Tried to access invalid coordinate vector index.");
	return m_v[i];
}

template <std::size_t dim, typename T>
T DistanceSq(const CoordVector<dim, T>& v1, const CoordVector<dim, T>& v2)
{
	T res = 0;
	for (std::size_t i = 0; i < dim; ++i){
		const T dist = v1[i] - v2[i];
		res += dist * dist;
	}
	return res;
}

template <std::size_t dim, typename T>
T Distance(const CoordVector<dim, T>& v1, const CoordVector<dim, T>& v2)
{
	return std::sqrt(DistanceSq(v1,v2));
}

template<std::size_t dim, typename T>
std::ostream& operator<<(std::ostream& stream, const CoordVector<dim, T>& v)
{
	stream << "(";
	for (std::size_t i = 0; i < dim-1; ++i)
		stream << v[i] << " ";
	stream << v[dim-1] << ")";

	return stream;
}

#endif // COORD_VECTOR_IMPL_H