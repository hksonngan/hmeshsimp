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
#include <stdlib.h>
#include <string>

#include <boost/unordered_set.hpp>

#include "lru_cache.h"
#include "ply_stream.h"
#include "util_common.h"

using std::ofstream;
using std::ostringstream;
using std::streampos;
using std::string;
using std::endl;

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

	/* ~ ply file read ~ */
	/* the first pass */
	bool readPlyFirst(const char* _ply_name);
	/* 
	 * the second pass
	 * X Y Z: x y z axis division count 
	 */
	bool readPlySecond(uint _X, uint _Y, uint _Z);
	
	char* info() const { return INFO_BUF; }
	/* set the temporary file directory */
	void tmpBase(const char *s) { tmp_base = s; }

private:
	/* ~ info ~ */
	inline void info(ostringstream &oss);
	inline void addInfo(const char *s);
	inline void addInfo(ostringstream &oss);
	inline void clearInfo();

	inline bool addVertexFirst(const int &i, const HVertex &v);

	/* ~ partitioning ~ */
	inline void getSlice();
	inline void getGrid(HVertex &v, HTripleIndex<uint> &grid_index);

private:
	/* a hash map, key is the grid coordinate, value is the index */
	HTripleIndexNumMap gridIndexMap;

	/* bound box */
	float		max_x, min_x; 
	float		max_y, min_y; 
	float		max_z, min_z;

	uint		x_div, y_div, z_div;
	float		x_slice, y_slice, z_slice;

	uint		vert_count;
	uint		face_count;

	char		*file_name;

	bool		binary_file;
	fpos_t		vert_start;
	char*		vertbin_name;
	LRUCache<LRUVertex>	
				vert_bin;
	/* the temporary file base directory */
	char*		tmp_base;
	LRUVertex	tmpv;

	char		INFO_BUF[INFO_BUF_SIZE];
	uint		info_buf_len;

	static const uint INFO_BUF_SIZE = 1000;
};


/* ========================== & IMPLEMENTATION & ======================= */

void HMeshGridDivide::info(ostringstream &oss) {

	clearInfo();
	addInfo(oss);
}

void HMeshGridDivide::addInfo(const char *s) {

	int len = strlen(s);
	memcpy(INFO_BUF + info_buf_len, s, len);
	info_buf_len += len;
	INFO_BUF[info_buf_len] = '\0';
}

void HMeshGridDivide::addInfo(ostringstream &oss) {

	addInfo(oss.str().c_str());
}

void HMeshGridDivide::clearInfo() {

	INFO_BUF[0] = '\0';
	info_buf_len += 0;
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
		tmpv.v = v;
		vert_bin.writeVal(tmpv);
	}

	return true;
}

void HMeshGridDivide::getSlice() {

	float _max_x = max_x;
	float _min_x = min_x;
	float _max_y = max_y;
	float _min_y = min_y;
	float _max_z = max_z;
	float _min_z = min_z;

	float half_range_x = (_max_x - _min_x) / 2 * 1.025;
	float half_range_y = (_max_y - _min_y) / 2 * 1.025;
	float half_range_z = (_max_z - _min_z) / 2 * 1.025;

	max_x = (_max_x + _min_x) / 2 + half_range_x;
	min_x = (_max_x + _min_x) / 2 - half_range_x;
	max_y = (_max_y + _min_y) / 2 + half_range_y;
	min_y = (_max_y + _min_y) / 2 - half_range_y;
	max_z = (_max_z + _min_z) / 2 + half_range_z;
	min_z = (_max_z + _min_z) / 2 - half_range_z;

	x_slice = (max_x - min_x) / x_div;
	y_slice = (max_y - min_y) / y_div;
	z_slice = (max_z - min_z) / z_div;
}

#endif //__H_DIVIDE_GRID_MESH__