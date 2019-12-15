/*
 * vector.h
 *
 * author: a.vogel@rub.de
 *
 */

#ifndef VECTOR_H
#define VECTOR_H

#include <vector>
#include <ostream>
#include <cstdint>

#include "parallel/layout.h"
#include "parallel/parallel_storage_type.h"


template <class T>
class Vector
 : private std::vector<T>
{
	public:
		/// entry type
		typedef T value_type;

		/// size type
		typedef std::size_t size_type;

	public:
		// method inherited from std::vector
	    using std::vector<T>::operator[];
    	using std::vector<T>::begin;
    	using std::vector<T>::end;
    	using std::vector<T>::size;
    	using std::vector<T>::resize;
    	using std::vector<T>::clear;

	public:
		/// constructor without arguments
		Vector(std::shared_ptr<AlgebraLayout> spLayouts = nullptr);

		/// constructor with size and optional default value
		Vector(std::size_t size, double val = 0.0, std::shared_ptr<AlgebraLayout> spLayouts = nullptr);

		/// set a constant value to each component
		Vector& operator=(double d);

		/// add a vector
		Vector& operator+=(const Vector& v);

		/// subtract a vector
		Vector& operator-=(const Vector& v);

		/// multiply by a scalar
		Vector& operator*=(double d);

		/// scalar product
		double operator*(const Vector& v) const;

		/// += s*v
		Vector& axpy(double s, const Vector& v); 

		/// euclidean norm
		double norm() const;

		/// set random values
		void random(T min, T max);

	public:
		/// set the storage type
		void set_storage_type(uint8_t type);

		/// change to requested storage type if possible
		bool change_storage_type(ParallelStorageType type);
		bool change_storage_type(ParallelStorageType type) const;

		/// return if storage type is contained in the actual storage type
		bool has_storage_type(uint8_t type) const;

		/// return the actual combination of storages types
		uint8_t get_storage_mask() const;

		/// return layout
		std::shared_ptr<AlgebraLayout> layouts() const {return m_spAlgebraLayout;}

		/// set layouts
		void set_layouts(std::shared_ptr<AlgebraLayout> spLayouts) {m_spAlgebraLayout = spLayouts;}

	protected:
		inline void check_size_matches(const Vector& v) const;

		///	add all slave values to master value
		void add_slaves_to_master();

		///	copy master value to all slaves
		void copy_master_to_slaves();

		/// set all slaves to zero
		void set_slaves_zero();

	protected:
		/// storage type
		uint8_t m_storageMask;

		/// algebra layouts
		std::shared_ptr<AlgebraLayout> m_spAlgebraLayout;		
};

/// output for vector
template <typename T>
std::ostream& operator<<(std::ostream& stream, const Vector<T>& v);

#ifdef CUDA
    #include "vector_impl_cuda.h"
#else
	#include "vector_impl.h"
#endif


#endif // VECTOR_H
