/*
 * ell_matrix.h
 *
 * author: a.vogel@rub.de
 *
 */

#ifndef ELL_MATRIX_MPI_H
#define ELL_MATRIX_MPI_H

#include <vector>
#include <ostream>
#include <type_traits>
#include <cassert>
#include <iostream>
#include <stdexcept>

#include "lil_matrix.h"
#include "vector.h"

#include "parallel/layout.h"
#include "parallel/parallel_storage_type.h"


template <class T>
class ELLMatrix
{
	public:
		/// entry type
		typedef T value_type;

		/// size type
		typedef std::size_t size_type;

	public:
		ELLMatrix(std::shared_ptr<AlgebraLayout> spLayouts = nullptr)
			: m_storageMask(PST_UNDEFINED), m_spAlgebraLayout(spLayouts)
		{
			m_max_row_size = m_num_rows = m_num_cols = 0;
		};

		/// constructor 
		ELLMatrix(const LILMatrix<T>& mat, std::shared_ptr<AlgebraLayout> spLayouts = nullptr)
			: m_storageMask(PST_UNDEFINED), m_spAlgebraLayout(spLayouts)
		{
			m_max_row_size = mat.max_row_size();
			m_num_rows = mat.num_rows();
			m_num_cols = mat.num_cols();

			m_vValue.resize(m_num_rows * m_max_row_size, 0.0);
			m_vIndex.resize(m_num_rows * m_max_row_size, 0);

			// copy rows
			for(size_type r = 0; r < m_num_rows; ++r){
				int k = 0;
				auto itEnd = mat.end(r);
				for(auto it = mat.begin(r); it != itEnd; ++it, ++k){
					m_vValue[r * m_max_row_size + k] = it.value();
					m_vIndex[r * m_max_row_size + k] = it.index();
				}
			}
		}

		/// return number of rows
		size_type num_rows() const {return m_num_rows;}

		/// return number of columns
		size_type num_cols() const {return m_num_cols;}

		/// const entry access
		T operator()(size_type r, size_type c) const
		{
			assert(r < num_rows() && c < num_cols() && "Invalid index requested.");

			for(size_type k = 0; k < m_max_row_size; ++k){
				if(m_vIndex[r * m_max_row_size + k] == c)
					return m_vValue[r * m_max_row_size + k];
			}
			return 0.0;
		}

		/// number of stored nonzeros
		size_type nnz() const {return m_vValue.size();}

		// max row size
		size_type max_row_size() const {return m_max_row_size;}

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

				friend class RowIterator<const value_type, const index_type>;
				friend class RowIterator<value_type, index_type>;

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

		row_iterator begin(size_type row)		{return row_iterator(&m_vValue[row * m_max_row_size], &m_vIndex[row * m_max_row_size], 0);}
		row_iterator end(size_type row)			{return row_iterator(&m_vValue[row * m_max_row_size], &m_vIndex[row * m_max_row_size], m_max_row_size);}

		const_row_iterator begin(size_type row) const {return const_row_iterator(&m_vValue[row * m_max_row_size], &m_vIndex[row * m_max_row_size], 0);}
		const_row_iterator end(size_type row) 	const {return const_row_iterator(&m_vValue[row * m_max_row_size], &m_vIndex[row * m_max_row_size], m_max_row_size);}

	public:
		/// set the storage type
		void set_storage_type(uint8_t type)
		{
			if(type == PST_UNIQUE) type = PST_UNIQUE | PST_ADDITIVE;
			m_storageMask = type;
		}

		/// return if storage type is contained in the actual storage type
		bool has_storage_type(uint8_t type) const
		{
			return type == PST_UNDEFINED ? m_storageMask == PST_UNDEFINED : (m_storageMask & type) == type;
		}

		/// return the actual combination of storages types
		uint8_t get_storage_mask() const
		{
			return m_storageMask;
		}

		/// return layout
		std::shared_ptr<AlgebraLayout> layouts() const {return m_spAlgebraLayout;}

		/// set layouts
		void set_layouts(std::shared_ptr<AlgebraLayout> spLayouts) {m_spAlgebraLayout = spLayouts;}

	protected:
		/// storage type
		uint8_t m_storageMask;

		/// algebra layouts
		std::shared_ptr<AlgebraLayout> m_spAlgebraLayout;		

	protected:
		// data storage
		size_type m_num_rows, m_num_cols, m_max_row_size;

		// Ellpack storage for entries and indices 
		std::vector<T> m_vValue;
		std::vector<size_type> m_vIndex;
};

/// output for dense matrices
template <typename T>
std::ostream& operator<<(std::ostream& stream, const ELLMatrix<T>& m)
{

	if (!m.num_cols()) return stream << "[]" << std::endl;

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


////////////////////////////////////////
// some help methods
////////////////////////////////////////


/// r = A*x
inline void Apply(Vector<double>& r,  const ELLMatrix<double>& A, const Vector<double>& x){

	
	if(!x.has_storage_type(PST_CONSISTENT))
		throw(std::runtime_error("Solution expected to be consistent"));

	if(x.size() != A.num_cols()){
		throw(std::runtime_error("Matrix columns must match Vector size."));
	}

	r.resize(A.num_rows());
	for(size_t i = 0; i < A.num_rows(); ++i){
		r[i] = 0.0;
		for(auto it = A.begin(i); it !=  A.end(i); ++it){
			r[i] += it.value() * x[ it.index() ];
		}
	}

	r.set_storage_type(PST_ADDITIVE);

}

/// r = A^T * x
inline void ApplyTransposed(Vector<double>& r,  const ELLMatrix<double>& A, const Vector<double>& x){

	if(x.size() != A.num_rows()){
		throw(std::runtime_error("Matrix columns must match Vector size."));
	}

	if(r.size() != A.num_cols()){
		throw(std::runtime_error("Matrix columns must match Vector size."));
	}

	for(size_t i = 0; i < A.num_cols(); ++i){
		r[i] = 0.0;
	}

	for(size_t i = 0; i < A.num_rows(); ++i){
		for(auto it = A.begin(i); it !=  A.end(i); ++it){
			r[ it.index() ] += it.value() * x[ i ];
		}
	}

	r.set_storage_type(PST_ADDITIVE);
}


/// r = r - A*x
inline void UpdateResiduum(Vector<double>& r,  const ELLMatrix<double>& A, const Vector<double>& x){

	if(!r.has_storage_type(PST_ADDITIVE))
		throw(std::runtime_error("Residuum expected to be additive"));

	if(!x.has_storage_type(PST_CONSISTENT))
		throw(std::runtime_error("solution expected to be consistent"));

	if(x.size() != A.num_cols())
		throw(std::runtime_error("Matrix columns must match Vector size."));

	r.resize(A.num_rows());
	for(size_t i = 0; i < A.num_rows(); ++i){
		for(auto it = A.begin(i); it !=  A.end(i); ++it){
			r[i] -= it.value() * x[ it.index() ];
		}
	}

	r.set_storage_type(PST_ADDITIVE);
}



/// r = b - A*x
inline void ComputeResiduum(Vector<double>& r,  const Vector<double>& b, const ELLMatrix<double>& A, const Vector<double>& x){

	if(!b.has_storage_type(PST_ADDITIVE))
		throw(std::runtime_error("Residuum expected to be additive"));

	if(!x.has_storage_type(PST_CONSISTENT))
		throw(std::runtime_error("Solution expected to be consistent"));

	if(x.size() != A.num_cols()){
		throw(std::runtime_error("Matrix columns must match Vector size."));
	}

	r.resize(A.num_rows());
	for(size_t i = 0; i < A.num_rows(); ++i){
		r[i] = b[i];
		for(auto it = A.begin(i); it !=  A.end(i); ++it){
			r[i] -= it.value() * x[ it.index() ];
		}
	}

	r.set_storage_type(PST_ADDITIVE);
}


/// computes P^T * A * P
inline void MultiplyPTAP(ELLMatrix<double>& PTAP,  const ELLMatrix<double>& A, const ELLMatrix<double>& P){

	if( A.num_cols() != P.num_rows() || P.num_rows() != A.num_rows() ){
		std::cout << "A: " << A.num_rows() << " x " << A.num_cols() << std::endl;
		std::cout << "P: " << P.num_rows() << " x " << P.num_cols() << std::endl;
		throw(std::runtime_error("Matrix sizes must be adequat."));
	}

	// todo: handle layouts and storage types correctly

	LILMatrix<double> RAP( P.num_cols(), P.num_cols() );

	// compute P^T A P = \sum P_ki * A_kl * P_lj
	for(size_t k = 0; k < P.num_rows(); ++k){
		for(auto it = P.begin(k); it !=  P.end(k); ++it){

			const size_t i = it.index();
			const double& P_ki = it.value();

			for(auto it = A.begin(k); it !=  A.end(k); ++it){

				const size_t l = it.index();
				const double& A_kl = it.value();

				const double P_ki_A_kl = P_ki * A_kl;

				for(auto it = P.begin(l); it !=  P.end(l); ++it){

					const size_t j = it.index();
					const double& P_lj = it.value();

					const double P_ki_A_kl_P_lj =  P_ki_A_kl * P_lj;
					if(P_ki_A_kl_P_lj != 0.0)
						RAP(i,j) += P_ki_A_kl_P_lj;
				}
			}
		}
	}

	PTAP = RAP;
}

#endif // ELL_MATRIX_MPI_H
