/*
 * parallel_storage_type.h
 *
 * author: a.vogel@rub.de
 *
 */

#ifndef PARALLEL_STORAGE_TYPE_H
#define PARALLEL_STORAGE_TYPE_H

#include <vector>
#include <ostream>
#include <cstdint>

#include "parallel/layout.h"


/// Parallel Storage type (see github.com/ug4)
/**
 * The storage type of a vector is used in parallel applications.
 * We assume that the dofs are distributed to the processes in the way that
 * each dof is master on exactly one process and can be a slave (i.e. a local
 * copy) on several other processes. Given the real values of the dofs the
 * different storage type are defined as follows:
 *  - PST_UNDEFINED: no information given
 *  - PST_CONSISTENT: The real value is saved in the master and every slave
 *  - PST_ADDITIVE: The sum over the values in the master and all slaves gives the exact value
 *  - PST_UNIQUE: Same as PST_ADDITIV, but value is zero in all slaves (i.e. master has exact value)
 *
 *  Note, that a Vector can have more than one type. E.g., every unique vector
 *  is additive. Moreover, the vector being zero everywhere is consistent,
 *  additive and unique at the same time. Therefore, the information is given
 *  bitwise.
 */
enum ParallelStorageType
{
	PST_UNDEFINED = 0,
	PST_CONSISTENT = 1 << 0,
	PST_ADDITIVE = 1 << 1,
	PST_UNIQUE = 1 << 2
};

// bitwise and for Parallel Storage Type
inline ParallelStorageType operator & (const ParallelStorageType &a, const ParallelStorageType &b)
{
	return (ParallelStorageType) ((int)a&(int)b);
}

inline std::ostream& operator<< (std::ostream& outStream, const ParallelStorageType& type)
{
	if(!type) outStream << "undefined";
	if(type & PST_CONSISTENT) outStream << "consistent";
	if(type & PST_UNIQUE) outStream << "unique";
	else if (type & PST_ADDITIVE) outStream << "additive";
	return outStream;
}


#endif // PARALLEL_STORAGE_TYPE_H
