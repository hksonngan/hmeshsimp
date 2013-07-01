/*
 *  Adaptively Generate the Iso-surfaces in Parallel
 *  The including file for host-side c++ invoking
 *
 *  Author: Ht
 *  Email : waytofall916 at gmail dot com
 *
 *  Copyright (C) Ht. All rights reserved.
 */

#include <string>

#define ___OUT

using std::string;

bool pAdaptiveIso(
	const string& filename, const int startDepth, const float errorThresh, 
	const float isovalue, ___OUT string& info);