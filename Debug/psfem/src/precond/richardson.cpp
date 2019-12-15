/*
 * richardson.cpp
 *
 * author: a.vogel@rub.de
 *
 */

#include <cstddef>
#include <cassert>
#include <iostream>

#include "richardson.h"


bool Richardson::precond(Vector<double>& c, const Vector<double>& r) const
{
	assert(c.size() == r.size() && "Size mismatch.");

	for (std::size_t i = 0; i < r.size(); ++i)
	{
        c[i] = m_eta * r[i];
	}

	c.set_storage_type(r.get_storage_mask());

	return true;
}

