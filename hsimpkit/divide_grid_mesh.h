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

#include <fstream>
#include <iostream>
#include <sstream>
#include <stdio.h>
#include <string>

#include <boost/unordered_set.hpp>

#include "lru_cache.h"
#include "ply_stream.h"
#include "util_common.h"

using std::ofstream;
using std::ostringstream;
using std::streampos;
using std::string;

/*
 *  The dividing algorithm takes two phase:
 *  1. retrieve the bounding box and create the
 *     vertex binary file (if the input file is
 *     text file) or convert add vertices coordinates
 *     to the face record.
 *  2. partition the mesh to patches
 */

/* ========================== & DEFINITION & ======================= */

/* map between grid index and the id of the patches */
typedef boost::unordered::unordered_map<HTripleIndex<uint>, unsigned int, HTripleIndexHash, HTripleIndexEqual> HTripleIndexNumMap;

/* out-of-core mesh divide base on the uniform grid */
class HMeshGridDivide {
public:
	HMeshGridDivide() {
		vertbin_name = NULL;
		tmp_base = NULL;
		INFO_BUF[0] = '\0';
		info_buf_len = 0;
	}
	~HMeshGridDivide() {
		if (vertbin_name)
			delete[] vertbin_name;
	}

	/* X Y Z: x y z axis division count */
	bool readPly(char* ply_name, uint X, uint Y, uint Z);
	
	char* info() const { return INFO_BUF; }
	void tmpBase(const char *s) { tmp_base = s; }

private:
	inline void info(ostringstream &oss);
	inline void addInfo(char *s);
	inline void addInfo(ostringstream &oss);
	inline void clearInfo();
	inline bool addVertexFirst(const int &i, const HVertex &v);

private:
	/* a hash map, key is the grid coordinate, value is the index */
	HTripleIndexNumMap gridIndexMap;
	/* bound box */
	float		max_x, min_x; 
	float		max_y, min_y; 
	float		max_z, min_z;

	bool		binary_file;
	ofstream	vertbin_out;
	fpos_t		vert_start;
	char*		vertbin_name;
	/* the temporary file base directory */
	char*		tmp_base;

	char		INFO_BUF[INFO_BUF_SIZE];
	uint		info_buf_len;

	static const uint INFO_BUF_SIZE = 1000;
};


/* ========================== & IMPLEMENTATION & ======================= */

void HMeshGridDivide::addInfo(char *s) {

}

void HMeshGridDivide::addInfo(ostringstream &oss) {

}

void HMeshGridDivide::clearInfo() {

}

bool HMeshGridDivide::addVertexFirst(const int &i, const HVertex &v) {
	
	if (i == 0) {
		max_x = min_x = v.x;
		max_y = min_y = v.y;
		max_z = min_z = v.z;
	}
	else {
		if (max_x < v.x)
			max_x = v.x;
		else if (min_x > v.x)
			min_x = v.x;

		if (max_y < v.y)
			max_y = v.y;
		else if (min_y > v.y)
			min_y = v.y;

		if (max_z < v.z)
			max_z = v.z;
		else if (min_z > v.z)
			min_z = v.z;
	}

	if (!binary_file) {
		/// to do!!
	}
}

#endif //__H_DIVIDE_GRID_MESH__