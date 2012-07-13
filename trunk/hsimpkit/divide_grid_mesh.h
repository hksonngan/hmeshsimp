/*
 *  Divide the mesh based on the uniform grid
 *
 *  Author: Ht
 *  Email : waytofall916@gmail.com
 *
 *  Copyright (C) Ht-waytofall. All rights reserved.
 *	
 *  This file is part of hmeshsimp.
 *
 *  hmeshsimp is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  hmeshsimp is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with hmeshsimp.  If not, see <http://www.gnu.org/licenses/>.
 */


#include "lru_cache.h"
#include "ply_stream.h"
#include <boost/unordered_set.hpp>

/* map between HTripleIndex<Integer> and the index of the patches */
typedef boost::unordered::unordered_map<HTripleIndex<Integer>, unsigned int, HTripleIndexHash, HTripleIndexEqual> HTripleIndexNumMap;

/* out-of-core mesh divide base on the uniform grid */
class MeshGridDivide {
public:
	/* X Y Z: x y z axis division count */
	bool readPly() (unsigned int X, unsigned int Y, unsigned int Z, char* ply_name);
	
private:
	bool addVertex(HVertex v); 
	bool getBoundbox();

private:
	// a hash map, key is the grid coordinate, value is the index
	HTripleIndexNumMap gridIndexMap;
	/* bound box */
	float max_x, min_x; float max_y, min_y; float max_z, min_z;
};

bool MeshGridDivide::readPly() (unsigned int X, unsigned int Y, unsigned int Z, char* ply_name) {
	
}