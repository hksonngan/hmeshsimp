/*
 *  A stream similar to fstream except it is 
 *  in-core and only handles T type data
 *
 *  Author: Ht
 *  Email : waytofall916 at gmail dot com
 *
 *  Copyright (C) Ht-waytofall. All rights reserved.
 */

#ifndef __H_MEM_STREAM__
#define __H_MEM_STREAM__

#include "h_dynamarray.h"

template<class T>
class mstream: public HDynamArray<T> {
public:
	mstream<T>& operator<< (T e) { 
		push_back(e); 
		return *this; }

	bool add(T e) { 
		push_back(e); 
		return true; }

	bool good() { return true; }
};

#endif //__H_MEM_STREAM__
