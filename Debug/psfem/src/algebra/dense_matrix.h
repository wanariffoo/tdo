/*
 * dense_matrix.h
 *
 * author: a.vogel@rub.de
 *
*/

#ifndef DENSE_MATRIX_H
#define DENSE_MATRIX_H

#include <vector>
#include <ostream>
#include <type_traits>
#include <cassert>

#include "lil_matrix.h"

template <class T>
class DenseMatrix
{
	public:
		/// entry type
		typedef T value_type;

		/// size type
		typedef std::size_t size_type;

	public:
		/// constructor without arguments
		DenseMatrix() : m_num_rows(0), m_num_cols(0) {}

		/// brief Constructor with size and optional default value
		DenseMatrix(size_type r, size_type c, T val = 0.0) {reinit(r, c, val);}

		/// constructor 
		DenseMatrix(const LILMatrix<T>& mat) {
			reinit(mat.num_rows(), mat.num_cols(), 0.0);

			// copy rows
			for(size_type r = 0; r < m_num_rows; ++r){
				auto itEnd = mat.end(r);
				for(auto it = mat.begin(r); it != itEnd; ++it){
					(*this)(r, it.index()) = it.value();
				}
			}
		}


		/// return number of rows
		size_type num_rows() const {return m_num_rows;}

		/// return number of columns
		size_type num_cols() const {return m_num_cols;}

		/// Resize the matrix
		void reinit(size_type r, size_type c, T val = 0.0){
			m_num_rows = r; m_num_cols = c;
			m_vValue.clear();
			m_vValue.resize(m_num_rows * m_num_cols, val);
		}

		/// const entry access
		const T& operator()(size_type r, size_type c) const
		{
			assert(r < num_rows() && c < num_cols() && "Invalid index requested.");
			return m_vValue[r * m_num_cols + c];
		}

		/// non-const entry access
		T& operator()(size_type r, size_type c)
		{
			assert(r < num_rows() && c < num_cols() && "Invalid index requested.");
			return m_vValue[ r * m_num_cols + c ];
		}

		/// number of stored nonzeros
		size_type nnz() const{ return m_vValue.size();}

	public:
		template <class ValueType>
		class RowIterator 
		{
			public:
				typedef std::bidirectional_iterator_tag  iterator_category;
				typedef typename std::remove_const<ValueType>::type	value_type; // mutable value type required for std::iterator_traits
				typedef std::ptrdiff_t	difference_type;
				typedef ValueType*		pointer;		// possibly const value type required for std::iterator_traits
				typedef ValueType&		reference;		// possibly const value type required for std::iterator_traits

				// friend class RowIterator<const value_type>;
				// friend class RowIterator<value_type>;

			public:
				RowIterator() : p(0), j(0) {}
				RowIterator(ValueType* _p, size_type _j) : 	p(_p), j(_j) {}
				RowIterator(const RowIterator<value_type>& o) : p(o.p), j(o.j) {}

				bool operator!=(const RowIterator<const value_type>& o) const {return j != o.j || p != o.p;}
				bool operator==(const RowIterator<const value_type>& o) const {return j == o.j && p == o.p;}

				RowIterator& operator++() {++j; return *this;}
				RowIterator& operator++(int) {RowIterator tmp(*this); ++(*this); return tmp;}

				RowIterator& operator--() {--j; return *this;}
				RowIterator& operator--(int) {RowIterator tmp(*this); --(*this); return tmp;}

				ValueType& value() const {return p[j];}
				ValueType& operator*() const {return value();}
				ValueType* operator->() const {return &value();}

				inline size_type index() const {return j;}

			protected:
				ValueType* p;
				size_type j;
		};

		typedef RowIterator<T> row_iterator;
		typedef RowIterator<const T> const_row_iterator;

		row_iterator begin(size_type row)		{return row_iterator(&m_vValue[ row * m_num_cols ], 0);}
		row_iterator end(size_type row)			{return row_iterator(&m_vValue[ row * m_num_cols ], m_num_cols);}

		const_row_iterator begin(size_type row) const {return const_row_iterator(&m_vValue[ row * m_num_cols ], 0);}
		const_row_iterator end(size_type row) 	const {return const_row_iterator(&m_vValue[ row * m_num_cols ], m_num_cols);}

	protected:
		// data storage
		size_type m_num_rows, m_num_cols;
		std::vector<T> m_vValue;
};

/// output for dense matrices
template <typename T>
std::ostream& operator<<(std::ostream& stream, const DenseMatrix<T>& m)
{

	if (!m.num_cols()) return stream << "[]";

	for (std::size_t i = 0; i < m.num_rows(); ++i){
		stream << "[";
		std::size_t sz = m.num_cols() - 1;
		for (std::size_t j = 0; j < sz; ++j){
			if(m(i,j) >= 0) stream << " ";
			stream << m(i,j) << " ";
		}
		if(m(i,sz) >= 0) stream << " ";
		stream << m(i,sz) << "]";
		stream << std::endl;
	}

	return stream;
}


#endif // DENSE_MATRIX_H
