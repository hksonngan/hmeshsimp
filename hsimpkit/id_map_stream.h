/*
 *  The Input-Output Id Map Class
 *
 *  Author: Ht
 *  Email : waytofall916 at gmail dot com
 *
 *  Copyright (C) Ht. All rights reserved.
 */

#ifndef __H_ID_MAP_STREAM__
#define __H_ID_MAP_STREAM__

#include "common_def.h"

// The Input-Output Id Map Class
// It is an in-core version
// An out-of-core version can be implemented 
// and used as an template type parameter in
// the simplification code
class IdMapMStream {
public:
	bool add(uint orig_id, uint new_id) {
		map[orig_id] = new_id;
		return true;
	}

	uint& operator[] (uint i) { return map[i]; }

public:
	uint_map map;
};

#endif //__H_ID_MAP_STREAM__