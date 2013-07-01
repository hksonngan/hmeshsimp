/*
 *  A Class That Defines a Vertex Used in LRU Cache
 *
 *  Author: Ht
 *  Email : waytofall916 at gmail dot com
 *
 *  Copyright (C) Ht. All rights reserved.
 */

#ifndef __LRU_VERTEX_H__
#define __LRU_VERTEX_H__

#include <iostream>
#include <stdio.h>
#include "common_def.h"
#include "common_types.h"
#include "io_common.h"

// A Class That Defines a Vertex Used in LRU Cache
class LRUVertex {
public:
	// binary(!) read
	bool read(ifstream& fin) {
		READ_BLOCK_BIN(fin, v.x, VERT_ITEM_SIZE);
		READ_BLOCK_BIN(fin, v.y, VERT_ITEM_SIZE);
		READ_BLOCK_BIN(fin, v.z, VERT_ITEM_SIZE);

		if (fin.good())
			return true;
		return false; 
	}

	bool read(FILE *fp) { 
		if (C_READ_BLOCK(fp, v.x, VERT_ITEM_SIZE, 1) != 1)
			return false;
		if (C_READ_BLOCK(fp, v.x, VERT_ITEM_SIZE, 1) != 1)
			return false;
		if (C_READ_BLOCK(fp, v.x, VERT_ITEM_SIZE, 1) != 1)
			return false;

		return true; 
	}

	// binary(!) write
	bool write(ofstream& fout) { 
		WRITE_BLOCK_BIN(fout, v.x, VERT_ITEM_SIZE);
		WRITE_BLOCK_BIN(fout, v.y, VERT_ITEM_SIZE);
		WRITE_BLOCK_BIN(fout, v.z, VERT_ITEM_SIZE);

		if (fout.good())
			return true;
		return false; 
	}

	// hash the index
	static unsigned int hash(unsigned int index) { return index; }

	// the size of the 
	static size_t size() { return sizeof(HVertex); }

public:
	HVertex v;
};

typedef LRUCache<LRUVertex> VertexBinary;

#endif