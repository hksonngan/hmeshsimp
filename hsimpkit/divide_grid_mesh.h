/*
 *  Divide the mesh based on the uniform grid
 *
 *  Author: Ht
 *  Email : waytofall916@gmail.com
 *
 *  Copyright (C) Ht-waytofall. All rights reserved.
 */

#ifndef __H_DIVIDE_GRID_MESH__
#define __H_DIVIDE_GRID_MESH__


#include <boost/unordered_set.hpp>
#include "lru_cache.h"
#include "ply_stream.h"
#include "util_common.h"


/* map between HTripleIndex<uint> and the index of the patches */
typedef boost::unordered::unordered_map<HTripleIndex<uint>, unsigned int, HTripleIndexHash, HTripleIndexEqual> HTripleIndexNumMap;

/* out-of-core mesh divide base on the uniform grid */
class HMeshGridDivide {
public:
	/* X Y Z: x y z axis division count */
	bool readPly() (char* ply_name, unsigned int X, unsigned int Y, unsigned int Z);
	bool addVertex(HVertex v);
	bool getBoundbox();

private:
	/* a hash map, key is the grid coordinate, value is the index */
	HTripleIndexNumMap gridIndexMap;
	/* bound box */
	float max_x, min_x; float max_y, min_y; float max_z, min_z;
};

bool HMeshGridDivide::readPly() (char* ply_name, unsigned int X, unsigned int Y, unsigned int Z) {
	
}

#endif //__H_DIVIDE_GRID_MESH__