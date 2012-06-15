/*
 * out of core vertex clustering algorithm run
 * author: ht
 * email : waytofall916@gmail.com
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