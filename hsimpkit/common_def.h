/*
 *  Common Definitions
 *
 *  Author: Ht
 *  Email : waytofall916 at gmail dot com
 *
 *  Copyright (C) Ht. All rights reserved.
 */

#ifndef __H_COMMON_DEF__
#define __H_COMMON_DEF__

#include <boost/unordered_map.hpp>

#define __IN
#define __OUT

typedef int integer;
typedef unsigned int uint;
typedef unsigned char uchar;

using boost::unordered::unordered_map;
typedef unordered_map<uint, uint> uint_map;

#define INVALID_CLUSTER_INDEX	UINT_MAX

#endif //__H_COMMON_DEF__