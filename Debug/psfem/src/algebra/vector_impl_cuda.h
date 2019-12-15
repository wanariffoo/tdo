

#ifndef VECTOR_IMPL_CUDA_H
#define VECTOR_IMPL_CUDA_H


#include "algebra/vector.h"
#include "parallel/parallel.h"
#include <cassert>
#include <cmath>
#include <ctime>
#include <iostream>


// Self-defined double-precision atomicAdd function for nvidia GPUs with Compute Capability 6 and below.
// Pre-defined atomicAdd() with double-precision does not work for pre-CC7 nvidia GPUs.
__device__ 
double atomicAdd_double(double* address, double val)
{
    unsigned long long int* address_as_ull =
                                (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val +
                                __longlong_as_double(assumed)));

    } while (assumed != old);

    return __longlong_as_double(old);
}

// Determines 1-dimensional CUDA block and grid sizes based on the number of rows N
__host__ 
void calculateDimensions(size_t N, dim3 &blockDim, dim3 &gridDim)
{
    if ( N <= 1024 )
    {
        blockDim.x = N; blockDim.y = 1; blockDim.z = 1;
        gridDim.x  = 1; gridDim.y = 1; gridDim.z = 1;
    }
        
    else
    {
        blockDim.x = 1024; blockDim.y = 1; blockDim.z = 1;
        gridDim.x  = (int)ceil(N/blockDim.x)+1; gridDim.y = 1; gridDim.z = 1;
    }
}

__global__
void setToZero(double* a, size_t num_rows)
{
	int id = blockDim.x * blockIdx.x + threadIdx.x;

	if ( id < num_rows )
		a[id] = 0.0;
}

// norm = x.norm()
__global__ 
void norm_GPU(double* norm, double* x, size_t num_rows)
{
	int id = blockDim.x * blockIdx.x + threadIdx.x;

	// TODO: if (id < num)

	if ( id == 0 )
		*norm = 0;
	__syncthreads();

	if ( id < num_rows )
	{
		atomicAdd_double( norm, x[id]*x[id] );
	}
	__syncthreads();

	if ( id == 0 )
		*norm = sqrt(*norm);
}

// a[] = 0


// a[] = 0, size_t
__global__
void setToZero(size_t* a, size_t num_rows)
{
	int id = blockDim.x * blockIdx.x + threadIdx.x;

	if ( id < num_rows )
		a[id] = 0.0;
}

//TODO: to delete
// bool = true
__global__
void setToTrue( bool *foo )
{
	*foo = true;
}


// DEBUG: TEST !!!!!!!!!!!!!!!!!!!!!!!!!!
__global__
void sqrt_GPU(double *x)
{
	*x = sqrt(*x);
}

// sum = sum( x[n]*x[n] )
__global__ 
void sumOfSquare_GPU(double* sum, double* x, size_t n)
{
	int id = threadIdx.x + blockDim.x*blockIdx.x;
	int stride = blockDim.x*gridDim.x;
		
	__shared__ double cache[1024];
	
	double temp = 0.0;
	while(id < n)
	{
		temp += x[id]*x[id];
		
		id += stride;
	}
	
	cache[threadIdx.x] = temp;
	
	__syncthreads();
	
	// reduction
	unsigned int i = blockDim.x/2;
	while(i != 0){
		if(threadIdx.x < i){
			cache[threadIdx.x] += cache[threadIdx.x + i];
		}
		__syncthreads();
		i /= 2;
	}

	// reset id
	id = threadIdx.x + blockDim.x*blockIdx.x;

	// reduce sum from all blocks' cache
	if(threadIdx.x == 0)
		atomicAdd_double(sum, cache[0]);
}


__global__ 
void LastBlockSumOfSquare_GPU(double* sum, double* x, size_t n, size_t counter)
{
	int id = threadIdx.x + blockDim.x*blockIdx.x;
    
    // if ( id >= counter*blockDim.x && id < ( ( counter*blockDim.x ) + lastBlockSize ) )
    if ( id >= counter*blockDim.x && id < n )
		atomicAdd_double(sum, x[id]*x[id]);
}

__host__
void norm_GPU(double* d_norm, double* d_x, size_t N, dim3 gridDim, dim3 blockDim)
{
	setToZero<<<1,1>>>( d_norm, 1);
    
    // getting the last block's size
    size_t lastBlockSize = N;
    size_t counter = 0;

    if ( N % gridDim.x == 0 ) {}
       

    else
    {
        while ( lastBlockSize >= gridDim.x)
        {
            counter++;
            lastBlockSize -= gridDim.x;
        }
    }

    // sum of squares for the full blocks
    // sumOfSquare_GPU<<<gridDim.x - 1, blockDim>>>(d_norm, d_x, N); // TODO: check, this is the original
    sumOfSquare_GPU<<<gridDim.x - 1, blockDim>>>(d_norm, d_x, (gridDim.x - 1)*blockDim.x);

    // sum of squares for the last incomplete block
    LastBlockSumOfSquare_GPU<<<1, lastBlockSize>>>(d_norm, d_x, N, counter);
	// cudaDeviceSynchronize();
	sqrt_GPU<<<1,1>>>( d_norm );
	// cudaDeviceSynchronize();
}


/// Helper functions for debugging
__global__ 
void print_GPU(double* x)
{
	printf("[GPU] x = %e\n", *x);
}

__global__ 
void print_GPU(int* x)
{
	printf("[GPU] x = %d\n", *x);
}

__global__ 
void print_GPU(size_t* x)
{
	printf("[GPU] x = %lu\n", *x);
}

__global__ 
void print_GPU(bool* x)
{
	printf("[GPU] x = %d\n", *x);
}

__global__ 
void printVector_GPU(double* x)
{
	int id = blockDim.x * blockIdx.x + threadIdx.x;

	printf("[GPU] x[%d] = %e\n", id, x[id]);
}

__global__
void printVector_GPU(double* x, size_t num_rows)
{
	int id = blockDim.x * blockIdx.x + threadIdx.x;

	if ( id < num_rows )
		printf("%d %e\n", id, x[id]);
}

__global__ 
void printVector_GPU(std::size_t* x)
{
	int id = blockDim.x * blockIdx.x + threadIdx.x;

	printf("[GPU] x[%d] = %lu\n", id, x[id]);
}

__global__ 
void printVector_GPU(int* x)
{
	int id = blockDim.x * blockIdx.x + threadIdx.x;

	printf("[GPU] x[%d] = %d\n", id, x[id]);
}

// (scalar) a = b
__global__ 
void equals_GPU(double* a, double* b)
{
	*a = *b;
}

// TODO: to delete this old dotproduct kernel
// x = a * b
// __global__ 
// void dotProduct(double* x, double* a, double* b, size_t num_rows)
// {
// 	int id = blockDim.x * blockIdx.x + threadIdx.x;

// 	if ( id == 0 )
// 		*x = 0;
// 	__syncthreads();

// 	if ( id < num_rows )
// 	{
// 		atomicAdd_double( x, a[id]*b[id] );
// 	}
// 	__syncthreads();
// }

// x = a * b
__global__ 
void dotProduct(double* x, double* a, double* b, size_t num_rows)
{
	unsigned int id = threadIdx.x + blockDim.x*blockIdx.x;
	unsigned int stride = blockDim.x*gridDim.x;

	__shared__ double cache[1024];

	double temp = 0.0;

	// filling in the shared variable
	while(id < num_rows){
		temp += a[id]*b[id];

		id += stride;
	}

	cache[threadIdx.x] = temp;

	__syncthreads();

	// reduction
	unsigned int i = blockDim.x/2;
	while(i != 0){
		if(threadIdx.x < i){
			cache[threadIdx.x] += cache[threadIdx.x + i];
		}
		__syncthreads();
		i /= 2;
	}

	if(threadIdx.x == 0){
		atomicAdd_double(x, cache[0]);
	}
	__syncthreads();
}


__global__
void LastBlockDotProduct(double* dot, double* x, double* y, size_t starting_index)
{
	int id = threadIdx.x + blockDim.x*blockIdx.x + starting_index;
		
	atomicAdd_double(dot, x[id]*y[id]);
	
}


// dot = a[] * b[]
__host__
void dotProduct_test(double* dot, double* a, double* b, size_t N, dim3 gridDim, dim3 blockDim)
{
    setToZero<<<1,1>>>( dot, 1 );

    // getting the last block's size
    size_t lastBlockSize = blockDim.x - ( (gridDim.x * blockDim.x ) - N );

	if ( N < blockDim.x)
	{
		LastBlockDotProduct<<<1, N>>>( dot, a, b, 0 );
	}

	else
	{
		// dot products for the full blocks
		dotProduct<<<gridDim.x - 1, blockDim>>>(dot, a, b, (gridDim.x - 1)*blockDim.x );
		
		// dot products for the last incomplete block
		LastBlockDotProduct<<<1, lastBlockSize>>>(dot, a, b, ( (gridDim.x - 1) * blockDim.x ) );
	}

}

// x = y / z
__global__
void divide_GPU(double *x, double *y, double *z)
{
	*x = *y / *z;
}
 

// TODO: doesn't work
// __global__ 
// void dotProduct(double* x, double* a, double* b, size_t num_rows)
// {	
// 	int index = threadIdx.x + blockIdx.x * blockDim.x;
// 	__shared__ double temp [1024];
	
// 	temp[threadIdx.x] = a[index] * b[index];
	
// 	__syncthreads(); 
	
// 	if( 0 == threadIdx.x ) 
// 	{
// 		double sum = 0;
// 		for( int i = 0; i < 1024; i++ )
// 		sum += temp[i];
// 		atomicAdd_double( x , sum );
// 	}
	
// }



// a = b
__global__ 
void vectorEquals_GPU(double* a, double* b, size_t num_rows)
{
	int id = blockDim.x * blockIdx.x + threadIdx.x;

	if ( id < num_rows )
		a[id] = b[id];
}


// x += c
__global__
void addVector_GPU(double *x, double *c, size_t num_rows)
{
	int id = blockDim.x * blockIdx.x + threadIdx.x;

	if ( id < num_rows )
		x[id] += c[id];
}

// x = x + alpha * p
__global__ 
void axpy_GPU(double* d_x, double* d_alpha, double* d_p, size_t num_rows)
{
	int id = blockDim.x * blockIdx.x + threadIdx.x;

	if ( id < num_rows )
		d_x[id] += (*d_alpha * d_p[id]);
}

// x = x - alpha * p
__global__ 
void axpy_neg_GPU(double* d_x, double* d_alpha, double* d_p, size_t num_rows)
{
	int id = blockDim.x * blockIdx.x + threadIdx.x;

	if ( id < num_rows )
		d_x[id] = d_x[id] - (*d_alpha * d_p[id]);
}

//TODO: doesn't work when >1 blocks
__global__ 
void sum_GPU(double* sum, double* x, size_t num_rows)
{
	int id = blockDim.x * blockIdx.x + threadIdx.x;

	if ( id == 0 )
		*sum = 0;
	__syncthreads();

	if ( id < num_rows )
	{
		atomicAdd_double( sum, x[id] );
	}

	__syncthreads();

	// if ( id == 0 )
	// 	printf("[GPU] : Sum = %e\n", *sum);
	
}



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
	change_storage_type(PST_UNIQUE);	// NOTE: i commented this out

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


#endif // VECTOR_IMPL_CUDA_H

