/*
	some global variables for hoocs
	author: ht
*/

#include "global_var.h"

//unsigned int g_x_partition = 10;
//unsigned int g_y_partition = 10;
//unsigned int g_z_partition = 10;
//
//char *g_infilename;
//
//BoundBox g_bound_box;
//
//char *g_tris_filename = NULL;
//
//unsigned char g_rcalc_policy;

HOOCSimp g_oocsimp;

#ifdef _DEBUG
std::ofstream dout ("debug_info.txt");
#endif