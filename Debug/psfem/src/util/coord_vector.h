/*
 * coord_vector.h
 *
 * author: a.vogel@rub.de
 */

#ifndef COORD_VECTOR_H
#define COORD_VECTOR_H

#include <cstddef>  // for std::size_t
#include <vector>   // for constructor with std::vector
#include <ostream>  // for output operator

/// A fixed size vector
template<std::size_t dim, typename T = double>
class CoordVector
{
	public:
        /// default constructor initiallizing to zero
        CoordVector();

        /// constructor initializing to given value
        CoordVector(T d);

        /// copy constructor
        CoordVector(const CoordVector<dim, T>& v);
    
        /// assignment operator
        CoordVector& operator=(const CoordVector<dim, T>& v);
    
        /// scale by constant
        CoordVector operator*(const T& s) const;

        /// const-access to element i
	const T& operator[](std::size_t i) const;

        /// non-const-access to element i
        T& operator[](std::size_t i);

	private:
        /// member data
	T m_v[dim];
};

/// returns the squared distance
template<std::size_t dim, typename T = double>
T Distance(const CoordVector<dim, T>& v1, const CoordVector<dim, T>& v2);

/// returns the squared distance
template<std::size_t dim, typename T = double>
T DistanceSq(const CoordVector<dim, T>& v1, const CoordVector<dim, T>& v2);

/// output for a vector
template<std::size_t dim, typename T>
std::ostream& operator<<(std::ostream& stream, const CoordVector<dim, T>& v);

// include implementation
#include "coord_vector_impl.h"

#endif // COORD_VECTOR_H
