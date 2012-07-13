/*
 *	Out-of-core algorithms implementation for hoocs
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

#ifndef __OOCSIMP__
#define __OOCSIMP__

#include "stdio.h"

typedef struct BoundBox
{
	float max_x, min_x;
	float max_y, min_y;
	float max_z, min_z;
	float center_x, center_y, center_z;
	float range;
} BoundBox;

class HOOCSimp
{
public:
	HOOCSimp() {
		tris_filename = NULL;
		cache_size = 500000;
	}

	~HOOCSimp() {
		if (tris_filename) {
			delete[] tris_filename;
		}
	}

	int oocsimp();
	int toTriangleSoup();

public:
	/* parameters for run oocsimp() */
	// partions of the grid of the bounding box
	unsigned int x_partition;
	unsigned int y_partition;
	unsigned int z_partition;
	unsigned char rcalc_policy; /* representative vertex calculating policy */
	char *infilename; /* input file name */
	unsigned int cache_size;

private:
	BoundBox bound_box;
	char *tris_filename; /* triangle soup file name */
};

#endif