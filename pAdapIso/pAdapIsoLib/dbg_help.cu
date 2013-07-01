/*
 *  Some Helper Functions for Debug
 *
 *  Author: Ht
 *  Email : waytofall916 at gmail dot com
 *
 *  Copyright (C) Ht. All rights reserved.
 */

#ifndef __DBG_HELP_CU__
#define __DBG_HELP_CU__

__device__ void printEdge(const OctEdge &edge) {
	char dir[3] = {'X', 'Y', 'Z'};

	CUPRINTF("edge, '%c', %hhd-%u %hhd-%u %hhd-%u %hhd-%u\n", dir[edge.edge_dir], 
		edge.level1, edge.index1, edge.level2, edge.index2, edge.level3, 
		edge.index3, edge.level4, edge.index4);
}

__device__ void printFace(const OctFace &face) {
	const char dir_str[3][3] = { "XY", "XZ", "YZ" };

	CUPRINTF("face, '%s', %hhd-%u %hhd-%u\n", dir_str[face.face_dir], 
		face.level1, face.index1, face.level2, face.index2);
}

#endif