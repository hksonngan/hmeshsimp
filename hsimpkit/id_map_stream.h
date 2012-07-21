/*
 *  
 *
 *  Author: Ht
 *  Email : waytofall916 at gmail dot com
 *
 *  Copyright (C) Ht-waytofall. All rights reserved.
 */


#ifndef __H_ID_MAP_STREAM__
#define __H_ID_MAP_STREAM__

#include "common_def.h"

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