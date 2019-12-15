/*
 * lil_matrix.h
 *
 * author: a.vogel@rub.de
 *
 */

#ifndef LIL_MATRIX_H
#define LIL_MATRIX_H

#include <vector>
#include <ostream>
#include <type_traits>
#include <cassert>
#include <cmath>

template <class T>
class LILMatrix
{
	public:
		/// entry type
		typedef T value_type;

		/// size type
		typedef std::size_t size_type;

	public:
		/// constructor without arguments
		LILMatrix() {reinit(0, 0);}

		/// constructor with size and optional default value
		LILMatrix(size_type r, size_type c, T val = 0.0) {reinit(r, c, val);}

		/// return number of rows
		size_type num_rows() const {return m_vvIndex.size();}

		/// return number of columns
		size_type num_cols() const {return m_num_cols;}

		/// Resize the matrix
		void reinit(size_type r, size_type c, T val = 0.0){
			m_vvValue.clear(); m_vvValue.resize(r);
			m_vvIndex.clear(); m_vvIndex.resize(r);
			m_num_cols = c;
		}

		/// non-const entry access
		T& operator()(size_type r, size_type c)
		{
			assert(r < num_rows() && c < num_cols() && "Invalid index requested.");

			for(size_type i = 0; i < m_vvIndex[r].size(); ++i){
				if(m_vvIndex[r][i] == c)
					return m_vvValue[r][i];
			}

			m_vvValue[r].push_back(0.0);
			m_vvIndex[r].push_back(c);
			return m_vvValue[r].back();
		}

		/// number of stored nonzeros
		size_type nnz() const{
			size_type nnz = 0;
			for(size_type r = 0; r < m_vvIndex.size(); ++r)
				nnz += m_vvIndex[r].size();
			return nnz;
		}

		/// number of stored nonzeros
		size_type max_row_size() const{
			using std::max;

			size_type mrz = 0;
			for(size_type r = 0; r < m_vvIndex.size(); ++r)
				mrz = max(mrz, m_vvIndex[r].size());
			return mrz;
		}

	public:
		template <class ValueType, class IndexType>
		class RowIterator 
		{
			public:
				typedef std::bidirectional_iterator_tag  iterator_category;
				typedef typename std::remove_const<ValueType>::type	value_type; // mutable value type required for std::iterator_traits
				typedef std::ptrdiff_t	difference_type;
				typedef ValueType*		pointer;		// possibly const value type required for std::iterator_traits
				typedef ValueType&		reference;		// possibly const value type required for std::iterator_traits
				typedef typename std::remove_const<IndexType>::type	index_type;	

	#ifndef CUDA
				friend class RowIterator<const value_type, const index_type>;
				friend class RowIterator<value_type, index_type>;
	#endif

			public:
				RowIterator() : p(0), ind(0), j(0) {}
				RowIterator(ValueType* _p, IndexType* _ind, size_type _j) : 	p(_p), ind(_ind), j(_j) {}
				RowIterator(const RowIterator<value_type,index_type>& o) : p(o.p), ind(o.ind), j(o.j) {}

				bool operator!=(const RowIterator<const value_type, const index_type>& o) const {return j != o.j || p != o.p;}
				bool operator==(const RowIterator<const value_type, const index_type>& o) const {return j == o.j && p == o.p;}

				RowIterator& operator++() {++j; return *this;}
				RowIterator& operator++(int) {RowIterator tmp(*this); ++(*this); return tmp;}

				RowIterator& operator--() {--j; return *this;}
				RowIterator& operator--(int) {RowIterator tmp(*this); --(*this); return tmp;}

				ValueType& value() const {return p[j];}
				ValueType& operator*() const {return value();}
				ValueType* operator->() const {return &value();}

				size_type index() const {return ind[j];}

			protected:
				ValueType* p;
				IndexType* ind;
				size_type j;
		};

		typedef RowIterator<T, size_type> row_iterator;
		typedef RowIterator<const T, const size_type> const_row_iterator;

		row_iterator begin(size_type row)		{return row_iterator(&m_vvValue[row][0], &m_vvIndex[row][0], 0);}
		row_iterator end(size_type row)			{return row_iterator(&m_vvValue[row][0], &m_vvIndex[row][0], m_vvValue[row].size());}

		const_row_iterator begin(size_type row) const {return const_row_iterator(&m_vvValue[row][0], &m_vvIndex[row][0], 0);}
		const_row_iterator end(size_type row) 	const {return const_row_iterator(&m_vvValue[row][0], &m_vvIndex[row][0], m_vvValue[row].size());}

	protected:
		// data storage
		size_type m_num_cols;

		// list of list of entry
		std::vector<std::vector<T>> m_vvValue;
		std::vector<std::vector<size_type>> m_vvIndex;
};

/// output for dense matrices
template <typename T>
std::ostream& operator<<(std::ostream& stream, const LILMatrix<T>& m)
{

	if (!m.num_cols()) return stream << "[]";

	for (std::size_t i = 0; i < m.num_rows(); ++i){
		stream << "[";
		auto iterEnd = m.end(i);		
		for (auto iter = m.begin(i); iter != iterEnd; ++iter){
			stream << "(" << iter.value() << "," << iter.index() << ")";
		}
		stream << "]";
		stream << std::endl;
	}

	return stream;
}


#endif // LIL_MATRIX_H
