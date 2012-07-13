/*
	some global variables for hoocs
	all variables plus g_ to denote
	that they are the globals
	author: ht
*/

#ifndef __HOOCS_GLOBAL_VAR__
#define __HOOCS_GLOBAL_VAR__

#include <iostream>
#include <fstream>
#include <string>
#include "vertex_cluster.h"
#include "oocsimp.h"

//typedef struct BoundBox
//{
//	float max_x, min_x;
//	float max_y, min_y;
//	float max_z, min_z;
//	float center_x, center_y, center_z;
//	float range;
//} BoundBox;
//
//// partions of the grid of the bounding box
//extern unsigned int g_x_partition;
//extern unsigned int g_y_partition;
//extern unsigned int g_z_partition;
//
//extern char *g_infilename; /* input file name */
//
//extern BoundBox g_bound_box;
//
//extern char *g_tris_filename; /* triangle soup file name */
//
//extern unsigned char g_rcalc_policy; /* representative vertex calculating policy */

extern HOOCSimp g_oocsimp;

#ifdef _DEBUG
extern std::ofstream fout;
#endif

#endif //__HOOCS_GLOBAL_VAR__