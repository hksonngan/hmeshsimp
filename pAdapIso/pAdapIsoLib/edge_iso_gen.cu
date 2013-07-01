/*
 *  Kernels of generating isosurfaces 
 *  from the minimal edges
 *
 *  Author: Ht
 *  Email : waytofall916 at gmail dot com
 *
 *  Copyright (C) Ht. All rights reserved.
 */
 
#ifndef _EDGE_ISO_GEN_KERNEL_H_
#define _EDGE_ISO_GEN_KERNEL_H_

typedef struct _VolEdge {
	XYZ p[2];
	float val[2];
} VolEdge;

// before reading the source codes, you can refer to figure at the beginning of 
// 'octree_edge.cu', and be familiar with the notion dimension 1, 2, 3 and node1
// node2, node3, node4 (node index) 

// get the coordinates of the dimension 2,3 for the node based on the node index
__forceinline__ __device__
void getEdgeDim23Coord(
	unsigned short &index2, unsigned short &index3, const unsigned short &len2, 
	const unsigned short &len3, const char &node_num_in_edge
){
	switch(node_num_in_edge) {
		case 1:
		index2 += len2;
		index3 += len3;
		break;
		case 2:
		index2 += len2;
		break;
		case 4:
		index3 += len3;
		break;
	}
}

// get the coordinates and volume value for endian points of an edge
__forceinline__ __device__
void getVolEdgeData(
	VolEdge &ve, const char *vol_data, unsigned short &index1, const unsigned short &len1,
	unsigned short &x, unsigned short &y, unsigned short &z 
){
	// note here!! the offset of voxel data must subtract the start coordinate
	getVoxelData(vol_data, ve.val[0], x-d_cube_start[d_max_depth[0]*3], 
		y-d_cube_start[d_max_depth[0]*3+1], z-d_cube_start[d_max_depth[0]*3+2]);
	ve.p[0].x = x*d_slice_thick[0];
	ve.p[0].y = y*d_slice_thick[1];
	ve.p[0].z = z*d_slice_thick[2];
	
	index1 += len1;

	// note here!! the offset of voxel data must subtract the start coordinate
	getVoxelData(vol_data, ve.val[1], x-d_cube_start[d_max_depth[0]*3], 
		y-d_cube_start[d_max_depth[0]*3+1], z-d_cube_start[d_max_depth[0]*3+2]);
	ve.p[1].x = x*d_slice_thick[0];
	ve.p[1].y = y*d_slice_thick[1];
	ve.p[1].z = z*d_slice_thick[2];
}

// get the coordinates and volume value for an edge based on a node and its index in 
// the edge
__forceinline__ __device__
void getNodeVolEdge(
	VolEdge &ve, const char *vol_data, const OctNode &node, const int &node_depth, 
	const unsigned char &edge_dir, const char node_num_in_edge
){
	unsigned short len_x, len_y, len_z;  // length of voxels in the finest depth
	unsigned short x, y, z;              // start of voxel offsets in the finest depth
	
	getNodeLenStart(node, node_depth, x, y, z, len_x, len_y, len_z);
	
	switch(edge_dir) {
	case X_AXIS:
		getEdgeDim23Coord(y, z, len_y, len_z, node_num_in_edge);
		getVolEdgeData(ve, vol_data, x, len_x, x, y, z);
		break;
	case Y_AXIS:
		getEdgeDim23Coord(x, z, len_x, len_z, node_num_in_edge);
		getVolEdgeData(ve, vol_data, y, len_y, x, y, z);
		break;
	case Z_AXIS:
		getEdgeDim23Coord(x, y, len_x, len_y, node_num_in_edge);
		getVolEdgeData(ve, vol_data, z, len_z, x, y, z);
		break;
	}
}

// configurable parameter
#define ONE_DIM_SAMPLE_COUNT 4
#define E 0.00001

template<typename DT, typename IT>
__forceinline__ __device__
void roundingDecimal(const DT &d, IT &i) {
	i = d;
	if (d-i > i+1-d)
		i ++;
}

// get the dual vertex for the face
__forceinline__ __device__
void getFaceDual(
	XYZW &face_dual, const unsigned char &face_dir, const char *vol_data,
	const OctNode &node1, const unsigned int &node_index1, const char &node_level1,
	const OctNode &node2, const unsigned int &node_index2, const char &node_level2
){
	char side; // side on which of the face the computing node is on 
	unsigned short len_x, len_y, len_z;       // length of voxels in the finest depth
	unsigned short start_x, start_y, start_z; // start of voxel offsets in the finest depth
	
	// level is greater means the node is in the lower level with a smaller cube
	if (node_index1 != INVALID_NODE && node_level1 >= node_level2) {
		side = 'l'; // l means the coordinate in the specific dimension of the node is less
		getNodeLenStart(
			node1, node_level1, start_x, start_y, start_z, len_x, len_y, len_z);
	} else if (node_index2 != INVALID_NODE) {
		side = 'r'; // r means the coordinate in the specific dimension of the node is greater
		getNodeLenStart(
			node2, node_level2, start_x, start_y, start_z, len_x, len_y, len_z);
	} else
		return;
	
	unsigned short end_x = start_x+len_x, end_y = start_y+len_y, end_z = start_z+len_z;
	
	switch(face_dir) {
	case XY_PLANE:
		if (side == 'r') {
			end_z = start_z;
		} else {
			start_z = end_z;
		}
		break;
	case XZ_PLANE:
		if (side == 'r') {
			end_y = start_y;
		} else {
			start_y = end_y;
		}
		break;
	case YZ_PLANE:
		if (side == 'r') {
			end_x = start_x;
		} else {
			start_x = end_x;
		}
		break;
	}
	
	float step_x = (float)len_x / (float)ONE_DIM_SAMPLE_COUNT;
	float step_y = (float)len_y / (float)ONE_DIM_SAMPLE_COUNT;
	float step_z = (float)len_z / (float)ONE_DIM_SAMPLE_COUNT;
	
	//unsigned short &x=len_x, &y=len_y, &z=len_z;
	unsigned short x, y, z;
	float float_x, float_y, float_z;

#ifdef __CUDA_DBG
	/////////////////////////////////
	// for debug !!
	//bool index_out_bound = false;
#endif
	
	char i = 0;
	float w;
	for (float_x = start_x; float_x <= (float)end_x/*+E*/; float_x += step_x) {
		for (float_y = start_y; float_y <= (float)end_y/*+E*/; float_y += step_y) {
			for (float_z = start_z; float_z <= (float)end_z/*+E*/; float_z += step_z) {
				roundingDecimal(float_x, x);
				roundingDecimal(float_y, y);
				roundingDecimal(float_z, z);

#ifdef __CUDA_DBG
				/////////////////////////////////
				// for debug !!
				//if (x < d_cube_start[d_max_depth[0]*3] || x > d_cube_start[d_max_depth[0]*3] + 
				//	d_cube_count[d_max_depth[0]*3] || 
				//	y < d_cube_start[d_max_depth[0]*3+1] || y > d_cube_start[d_max_depth[0]*3+1] + 
				//	d_cube_count[d_max_depth[0]*3+1] || 
				//	z < d_cube_start[d_max_depth[0]*3+2] || z > d_cube_start[d_max_depth[0]*3+2] + 
				//	d_cube_count[d_max_depth[0]*3+2]){

				//	index_out_bound = true;
				//	//CUPRINTF("x, y, z: %hu, %hu, %hu\n", x, y, z);
				//	//CUPRINTF("float x, y, z: %.2f, %.2f, %.2f\n", float_x, float_y, float_z);
				//	continue;
				//}
#endif
			
				// note here!! the offset of voxel data must subtract the start coordinate
				getVoxelData(vol_data, w, x-d_cube_start[d_max_depth[0]*3], 
					y-d_cube_start[d_max_depth[0]*3+1], z-d_cube_start[d_max_depth[0]*3+2]);
				face_dual.x = (float)x*d_slice_thick[0]/(float)(i+1) + face_dual.x*i/(float)(i+1);
				face_dual.y = (float)y*d_slice_thick[1]/(float)(i+1) + face_dual.y*i/(float)(i+1);
				face_dual.z = (float)z*d_slice_thick[2]/(float)(i+1) + face_dual.z*i/(float)(i+1);
				face_dual.w = w/(float)(i+1) + face_dual.w*i/(float)(i+1);
				i ++;
			}
		}
	}

#ifdef __CUDA_DBG
	//if (index_out_bound) {
	//	if (side == 'l') {
	//		CUPRINTF("node1: %hhd-%d, %hd %hd %hd\n", node_level1, node_index1, 
	//			node1.cube_index[0], node1.cube_index[1], node1.cube_index[2]);
	//	} else {
	//		CUPRINTF("node2: %hhd-%d, %hd %hd %hd\n", node_level2, node_index2, 
	//			node2.cube_index[0], node2.cube_index[1], node2.cube_index[2]);
	//	}
	//	
	//	CUPRINTF("start: %hu, %hu, %hu\n", start_x, start_y, start_z);
	//	CUPRINTF("end: %hu, %hu, %hu\n", end_x, end_y, end_z);
	//	//CUPRINTF("len: %hu, %hu, %hu\n", len_x, len_y, len_z);
	//	//CUPRINTF("step: %.2f, %.2f, %.2f\n", step_x, step_y, step_z);
	//}
#endif
}

__forceinline__ __device__
void assign (XYZ &a, float &w, const XYZW &b) {
	a.x = b.x;
	a.y = b.y;
	a.z = b.z;
	w = b.w;
}

// get the tetrahedron formed by each face
// the coordinate of the 'diff dimension' of node1 is less than node2
__forceinline__ __device__
void getFaceTetra(
	const VolEdge &ve, const unsigned char &face_dir, const char *vol_data,
	const OctNode &node1, const unsigned int &node_index1, const char &node_level1,
	const OctNode &node2, const unsigned int &node_index2, const char &node_level2,
	const OctNode &left_node, const unsigned int &left_index, 
	const OctNode &right_node, const unsigned int &right_index, 
	Tetra *tetra, char &tetra_count
){
	XYZW face_dual;
	
	getFaceDual(face_dual, face_dir, vol_data, node1, node_index1, node_level1, 
		node2, node_index2, node_level2);

#ifdef __CUDA_DBG
	//CUPRINTF("face dual, %.2f %.2f %.2f %.2f\n", face_dual.x, face_dual.y, face_dual.z, face_dual.w);
	//CUPRINTF("node, %hhd-%u, dual %.2f %.2f %.2f %.2f\n", node_level1, node_index1, 
	//	node1.dual_vert.x, node1.dual_vert.y, node1.dual_vert.z, node1.dual_vert.w);
	//CUPRINTF("node, %hhd-%u, dual %.2f %.2f %.2f %.2f\n", node_level2, node_index2, 
	//	node2.dual_vert.x, node2.dual_vert.y, node2.dual_vert.z, node2.dual_vert.w);
#endif
	
	tetra_count = 0;
	if (left_index != INVALID_NODE) {
		tetra[tetra_count].p[0] = ve.p[0];
		tetra[tetra_count].val[0] = ve.val[0];
		tetra[tetra_count].p[1] = ve.p[1];
		tetra[tetra_count].val[1] = ve.val[1];
		assign(tetra[tetra_count].p[3], tetra[tetra_count].val[3], left_node.dual_vert);
		assign(tetra[tetra_count].p[2], tetra[tetra_count].val[2], face_dual);	
		tetra_count ++;
	}
	if (right_index != INVALID_NODE) {
		tetra[tetra_count].p[0] = ve.p[0];
		tetra[tetra_count].val[0] = ve.val[0];
		tetra[tetra_count].p[1] = ve.p[1];
		tetra[tetra_count].val[1] = ve.val[1];
		assign(tetra[tetra_count].p[3], tetra[tetra_count].val[3], face_dual);
		assign(tetra[tetra_count].p[2], tetra[tetra_count].val[2], right_node.dual_vert);
		tetra_count ++;
	}
}

// get the tetrahedron formed by each edge
__forceinline__ __device__
void getEdgeTetra(
	const OctEdge &edge, const int &edge_depth, const char *vol_data, Tetra *tetra, 
	char &tetra_count
){
	VolEdge ve;
	OctNode node1, node2, node3, node4;
	bool vol_edge_retrieved = false;

#ifdef __CUDA_DBG
	//printEdge(edge);
#endif

	if (edge.index1 != INVALID_NODE) {
		node1 = dev_octlvl_ptr[edge.level1][edge.index1];
		
		if (edge.level1 == edge_depth && !vol_edge_retrieved) {		
			getNodeVolEdge(ve, vol_data, node1, edge_depth, edge.edge_dir, 1);
			vol_edge_retrieved = true;
		}
	}
	if (edge.index2 != INVALID_NODE) {
		node2 = dev_octlvl_ptr[edge.level2][edge.index2];
		
		if (edge.level2 == edge_depth && !vol_edge_retrieved) {
			getNodeVolEdge(ve, vol_data, node2, edge_depth, edge.edge_dir, 2);
			vol_edge_retrieved = true;
		}
	}
	if (edge.index3 != INVALID_NODE) {
		node3 = dev_octlvl_ptr[edge.level3][edge.index3];
		
		if (edge.level3 == edge_depth && !vol_edge_retrieved) {
			getNodeVolEdge(ve, vol_data, node3, edge_depth, edge.edge_dir, 3);
			vol_edge_retrieved = true;
		}
	}
	if (edge.index4 != INVALID_NODE) {
		node4 = dev_octlvl_ptr[edge.level4][edge.index4];
		
		if (edge.level4 == edge_depth && !vol_edge_retrieved) {
			getNodeVolEdge(ve, vol_data, node4, edge_depth, edge.edge_dir, 4);
			vol_edge_retrieved = true;
		}
	}
	
	tetra_count = 0;
	char f_tetra_count = 0;
	switch(edge.edge_dir) {
	case X_AXIS:
		if (edge.index1 != edge.index2 || edge.level1 != edge.level2) {
			getFaceTetra(ve, XY_PLANE, vol_data, 
				node1, edge.index1, edge.level1, /* node 1 */
				node2, edge.index2, edge.level2, /* node 2 */
				node1, edge.index1,              /* left node */ 
				node2, edge.index2,              /* right node */
				tetra + tetra_count, f_tetra_count);
			tetra_count += f_tetra_count;
#ifdef __CUDA_DBG
			//CUPRINTF("f1 tetra count: %hhd\n", f_tetra_count);
#endif
		}
		if (edge.index2 != edge.index3 || edge.level2 != edge.level3) {
			getFaceTetra(ve, XZ_PLANE, vol_data, 
				node2, edge.index2, edge.level2,  /* node 1 */
				node3, edge.index3, edge.level3,  /* node 2 */
				node2, edge.index2,               /* left node */
				node3, edge.index3,               /* right node */
				tetra + tetra_count, f_tetra_count);
			tetra_count += f_tetra_count;
#ifdef __CUDA_DBG
			//CUPRINTF("f2 tetra count: %hhd\n", f_tetra_count);
#endif
		}
		if (edge.index3 != edge.index4 || edge.level3 != edge.level4) {
			getFaceTetra(ve, XY_PLANE, vol_data, 
				node4, edge.index4, edge.level4,  /* node 1 */
				node3, edge.index3, edge.level3,  /* node 2 */
				node3, edge.index3,               /* left node */
				node4, edge.index4,               /* right node */
				tetra + tetra_count, f_tetra_count);
			tetra_count += f_tetra_count;
#ifdef __CUDA_DBG
			//CUPRINTF("f3 tetra count: %hhd\n", f_tetra_count);
#endif
		}
		if (edge.index4 != edge.index1 || edge.level4 != edge.level1) {
			getFaceTetra(ve, XZ_PLANE, vol_data, 
				node1, edge.index1, edge.level1,  /* node 1 */
				node4, edge.index4, edge.level4,  /* node 2 */
				node4, edge.index4,               /* left node */
				node1, edge.index1,               /* right node */
				tetra + tetra_count, f_tetra_count);
			tetra_count += f_tetra_count;
#ifdef __CUDA_DBG
			//CUPRINTF("f4 tetra count: %hhd\n", f_tetra_count);
			//CUPRINTF("tetra count: %hhd\n", tetra_count);
#endif
		}
		break;
	case Y_AXIS:
		if (edge.index1 != edge.index2 || edge.level1 != edge.level2) {
			getFaceTetra(ve, XY_PLANE, vol_data, 
				node1, edge.index1, edge.level1,  /* node 1 */
				node2, edge.index2, edge.level2,  /* node 2 */
				node2, edge.index2,               /* left node */
				node1, edge.index1,               /* right node */
				tetra + tetra_count, f_tetra_count);
			tetra_count += f_tetra_count;
		}
		if (edge.index2 != edge.index3 || edge.level2 != edge.level3) {
			getFaceTetra(ve, YZ_PLANE, vol_data, 
				node2, edge.index2, edge.level2,  /* node 1 */
				node3, edge.index3, edge.level3,  /* node 2 */
				node3, edge.index3,               /* left node */
				node2, edge.index2,               /* right node */
				tetra + tetra_count, f_tetra_count);
			tetra_count += f_tetra_count;
		}
		if (edge.index3 != edge.index4 || edge.level3 != edge.level4) {
			getFaceTetra(ve, XY_PLANE, vol_data, 
				node4, edge.index4, edge.level4,  /* node 1 */
				node3, edge.index3, edge.level3,  /* node 2 */
				node4, edge.index4,               /* left node */
				node3, edge.index3,               /* right node */
				tetra + tetra_count, f_tetra_count);
			tetra_count += f_tetra_count;
		}
		if (edge.index4 != edge.index1 || edge.level4 != edge.level1) {
			getFaceTetra(ve, YZ_PLANE, vol_data, 
				node1, edge.index1, edge.level1,  /* node 1 */
				node4, edge.index4, edge.level4,  /* node 2 */
				node1, edge.index1,               /* left node */
				node4, edge.index4,               /* right node */
				tetra + tetra_count, f_tetra_count);
			tetra_count += f_tetra_count;
		}
		break;
	case Z_AXIS:
		if (edge.index1 != edge.index2 || edge.level1 != edge.level2) {
			getFaceTetra(ve, XZ_PLANE, vol_data, 
				node1, edge.index1, edge.level1,  /* node 1 */
				node2, edge.index2, edge.level2,  /* node 2 */
				node1, edge.index1,               /* left node */
				node2, edge.index2,               /* right node */
				tetra + tetra_count, f_tetra_count);
			tetra_count += f_tetra_count;
		}
		if (edge.index2 != edge.index3 || edge.level2 != edge.level3) {
			getFaceTetra(ve, YZ_PLANE, vol_data, 
				node2, edge.index2, edge.level2,  /* node 1 */
				node3, edge.index3, edge.level3,  /* node 2 */
				node2, edge.index2,               /* left node */
				node3, edge.index3,               /* right node */
				tetra + tetra_count, f_tetra_count);
			tetra_count += f_tetra_count;
		}
		if (edge.index3 != edge.index4 || edge.level3 != edge.level4) {
			getFaceTetra(ve, XZ_PLANE, vol_data, 
				node4, edge.index4, edge.level4,  /* node 1 */
				node3, edge.index3, edge.level3,  /* node 2 */
				node3, edge.index3,               /* left node */
				node4, edge.index4,               /* right node */
				tetra + tetra_count, f_tetra_count);
			tetra_count += f_tetra_count;
		}
		if (edge.index4 != edge.index1 || edge.level4 != edge.level1) {
			getFaceTetra(ve, YZ_PLANE, vol_data, 
				node1, edge.index1, edge.level1,  /* node 1 */
				node4, edge.index4, edge.level4,  /* node 2 */
				node4, edge.index4,               /* left node */
				node1, edge.index1,               /* right node */
				tetra + tetra_count, f_tetra_count);
			tetra_count += f_tetra_count;
		}
		break;
	}
}

// get the count of isosurfaces from each edge
__global__
void getEdgeIsosurfCountKn(
	const char* vol_data, const int start_depth, const float isovalue, 
	unsigned int *tri_count 
){
	const unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;

	////////////////////////
	// for debug !!
	//if (tid > 0)
	//	return;
	//tid = 104769;

	int depth;
	for (depth = start_depth; depth <= d_max_depth[0]; depth ++)
		if (tid < d_n_medge[depth])
			break;

	if (depth > d_max_depth[0])
		return;

	unsigned int eid = tid;
	if (depth > start_depth) {
		eid -= d_n_medge[depth-1];
	}

	const OctEdge edge = d_medge_ptr[depth][eid];

	unsigned char n_tri = 0;

#ifdef __CUDA_DBG
	// for debug !!
	//CUPRINTF("tetra count: %hhd\n", tetra_count);
	//char t = 0;
	//CUPRINTF("t%hhd\n", t);
	//if (tetra_count > t) {
	//	for (char i = 0; i < 4; i ++) {
	//		CUPRINTF("v%hhd, %.2f %.2f %.2f %.2f\n", i, tetra[t].p[i].x, 
	//			tetra[t].p[i].y, tetra[t].p[i].z, tetra[t].val[i]);
	//	}
	//}
#endif

	Tetra tetra[8];
	char tetra_count;
	getEdgeTetra(edge, depth, vol_data, tetra, tetra_count);

	for (char i = 0; i < tetra_count; i ++) {
		n_tri += PolygoniseTriGetCount(tetra[i], isovalue);
	}

	tri_count[tid] = n_tri;

#ifdef __CUDA_DBG
	// for debug !!
	//if (tid == 104769) {
	//	d_dbg_buf[0][0] = tetra_count;
	//	//for (char i = 0; i < tetra_count; i ++) {
	//	//	n_tri = PolygoniseTriGetCount(tetra[i], isovalue);
	//	//	d_dbg_buf[0][i+2] = n_tri;
	//	//}
	//	for (char i = 0; i < tetra_count; i ++) {
	//		for (char j = 0; j < 4; j ++) {
	//			d_dbg_buf[0][i*16+j*4+1] = tetra[i].p[j].x;
	//			d_dbg_buf[0][i*16+j*4+2] = tetra[i].p[j].y;
	//			d_dbg_buf[0][i*16+j*4+3] = tetra[i].p[j].z;
	//			d_dbg_buf[0][i*16+j*4+4] = tetra[i].val[j];
	//		}
	//	}
	//}
#endif
}

// get isosurfaces from each edge
__global__
void genEdgeIsosurfKn(
	const char* vol_data, const int start_depth, const float isovalue, 
	const unsigned int *tri_addr, float *tri 
){
	const unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;

	int depth;
	for (depth = start_depth; depth <= d_max_depth[0]; depth ++)
		if (tid < d_n_medge[depth])
			break;

	if (depth > d_max_depth[0])
		return;
	
	unsigned int addr = 0;
	unsigned int n_tri = 0;
	
	if (tid > 0)
		addr = tri_addr[tid-1];
	n_tri = tri_addr[tid];

	if (n_tri - addr <= 0)
		return;
		
	unsigned int eid = tid;
	if (depth > start_depth) {
		eid -= d_n_medge[depth-1];
	}

	const OctEdge edge = d_medge_ptr[depth][eid];

	Tetra tetra[8];
	char tetra_count;
	getEdgeTetra(edge, depth, vol_data, tetra, tetra_count);

	float ptri[18];
	n_tri = 0;

	char i, j;
	for (i = 0; i < tetra_count; i ++) {
		n_tri = PolygoniseTri(tetra[i], isovalue, ptri);
		for (j = 0; j < n_tri; j ++)
			copy9(tri+(addr+j)*9, ptri+j*9);
		addr += n_tri;
	}

#ifdef __CUDA_DBG
	// for debug !!
	//if (tid == 104769) {
	//	d_dbg_buf[0][0] = tetra_count;
	//	for (i = 0; i < tetra_count; i ++) {
	//		n_tri = PolygoniseTriGetCount(tetra[i], isovalue);
	//		d_dbg_buf[0][i+1] = n_tri;
	//	}
	//	for (i = 0; i < tetra_count; i ++) {
	//		for (j = 0; j < 4; j ++) {
	//			d_dbg_buf[0][i*16+j*4+1] = tetra[i].p[j].x;
	//			d_dbg_buf[0][i*16+j*4+2] = tetra[i].p[j].y;
	//			d_dbg_buf[0][i*16+j*4+3] = tetra[i].p[j].z;
	//			d_dbg_buf[0][i*16+j*4+4] = tetra[i].val[j];
	//		}
	//	}
	//}
#endif
}

#endif


/////////////////////////////
// for debug !! 
//unsigned short len_x, len_y, len_z, x, y, z;
//getNodeLenStart(node1, edge_depth, x, y, z, len_x, len_y, len_z);
//if (x < d_cube_start[d_max_depth[0]*3] || x > d_cube_start[d_max_depth[0]*3] + 
//	d_cube_count[d_max_depth[0]*3] || 
//	y < d_cube_start[d_max_depth[0]*3+1] || y > d_cube_start[d_max_depth[0]*3+1] + 
//	d_cube_count[d_max_depth[0]*3+1] || 
//	z < d_cube_start[d_max_depth[0]*3+2] || z > d_cube_start[d_max_depth[0]*3+2] + 
//	d_cube_count[d_max_depth[0]*3+2]) {
//	CUPRINTF("node index: %d %d %d\n", node1.cube_index[0], node1.cube_index[1], node1.cube_index[2]);
//	printEdge(edge);
//	CUPRINTF("error node num in edge: %d\n", 1);
//}
			

/////////////////////////////
// for debug !!
//unsigned short len_x, len_y, len_z, x, y, z;
//getNodeLenStart(node2, edge_depth, x, y, z, len_x, len_y, len_z);
//if (x < d_cube_start[d_max_depth[0]*3] || x > d_cube_start[d_max_depth[0]*3] + 
//	d_cube_count[d_max_depth[0]*3] || 
//	y < d_cube_start[d_max_depth[0]*3+1] || y > d_cube_start[d_max_depth[0]*3+1] + 
//	d_cube_count[d_max_depth[0]*3+1] || 
//	z < d_cube_start[d_max_depth[0]*3+2] || z > d_cube_start[d_max_depth[0]*3+2] + 
//	d_cube_count[d_max_depth[0]*3+2]) {
//	CUPRINTF("node index: %d %d %d\n", node2.cube_index[0], node2.cube_index[1], node2.cube_index[2]);
//	printEdge(edge);
//	CUPRINTF("error node num in edge: %d\n", 2);
//}


/////////////////////////////
// for debug !!
//unsigned short len_x, len_y, len_z, x, y, z;
//getNodeLenStart(node3, edge_depth, x, y, z, len_x, len_y, len_z);
//if (x < d_cube_start[d_max_depth[0]*3] || x > d_cube_start[d_max_depth[0]*3] + 
//	d_cube_count[d_max_depth[0]*3] || 
//	y < d_cube_start[d_max_depth[0]*3+1] || y > d_cube_start[d_max_depth[0]*3+1] + 
//	d_cube_count[d_max_depth[0]*3+1] || 
//	z < d_cube_start[d_max_depth[0]*3+2] || z > d_cube_start[d_max_depth[0]*3+2] + 
//	d_cube_count[d_max_depth[0]*3+2]) {
//	CUPRINTF("node index: %d %d %d\n", node3.cube_index[0], node3.cube_index[1], node3.cube_index[2]);
//	printEdge(edge);
//	CUPRINTF("error node num in edge: %d\n", 3);
//}
			
	
/////////////////////////////
// for debug !!
//unsigned short len_x, len_y, len_z, x, y, z;
//getNodeLenStart(node4, edge_depth, x, y, z, len_x, len_y, len_z);
//if (x < d_cube_start[d_max_depth[0]*3] || x > d_cube_start[d_max_depth[0]*3] + 
//	d_cube_count[d_max_depth[0]*3] || 
//	y < d_cube_start[d_max_depth[0]*3+1] || y > d_cube_start[d_max_depth[0]*3+1] + 
//	d_cube_count[d_max_depth[0]*3+1] || 
//	z < d_cube_start[d_max_depth[0]*3+2] || z > d_cube_start[d_max_depth[0]*3+2] + 
//	d_cube_count[d_max_depth[0]*3+2]) {
//	CUPRINTF("node index: %d %d %d\n", node4.cube_index[0], node4.cube_index[1], node4.cube_index[2]);
//	printEdge(edge);
//	CUPRINTF("error node num in edge: %d\n", 4);
//}
