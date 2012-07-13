/*
 *  Out of core vertex clustering algorithm run
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
#ifndef __OOC_VERTEX_CLUSTERING__
#define __OOC_VERTEX_CLUSTERING__

#include "tri_soup_stream.h"
#include "vertex_cluster.h"
#include <iostream>

class HOOCVertexClustering
{
public:
	bool run(int x_partition, int y_partition, int z_partition, RepCalcPolicy p,
		char* inputfilename, char* outputfilename);
};

#endif