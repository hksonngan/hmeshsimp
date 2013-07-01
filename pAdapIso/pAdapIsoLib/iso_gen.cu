/*
 *  Kernels of generating isosurfaces 
 *  from the minimal faces
 *
 *  Author: Ht
 *  Email : waytofall916 at gmail dot com
 *
 *  Copyright (C) Ht. All rights reserved.
 */

#ifndef _ISO_GEN_KERNEL_H_
#define _ISO_GEN_KERNEL_H_

#include "mt.cu"

/**************************************************
  about direction:
    the coordinate axis grow from left to right
    which is greater, right to left which is less
**************************************************/

template<typename T> 
__forceinline__ __device__ 
void getVoxelData(
    const char *vol_data, T &rvalue, const unsigned short &x, 
	const unsigned short &y, const unsigned short &z);

__device__
void getNodeLenStart(
	const OctNode &node, const int &depth, unsigned short &start_x, 
	unsigned short &start_y, unsigned short &start_z, unsigned short &len_x, 
	unsigned short &len_y, unsigned short &len_z
){
	// length in each dimension equals to the cube size if it's not on the side
	len_x = d_cube_size[depth];
	len_y = len_z = len_x;
	// note the start here is in fact the coordinate in the finest depth
	// but the result here is not for the endian one
	// for the start and end cude, the value should be recomputed
	start_x = (node.cube_index[0] + d_cube_start[depth*3]  ) * len_x;
	start_y = (node.cube_index[1] + d_cube_start[depth*3+1]) * len_y;
	start_z = (node.cube_index[2] + d_cube_start[depth*3+2]) * len_z;

	/////////////////////////////////////////////////////////////////
	// !! debug
	//if (threadIdx.x + blockIdx.x * blockDim.x == 580060) {
	//	d_dbg_buf[0][0] = node.cube_index[0];
	//	d_dbg_buf[0][1] = node.cube_index[1];
	//	d_dbg_buf[0][2] = node.cube_index[2];
	//}
	/////////////////////////////////////////////////////////////////

	if (depth != d_max_depth[0]) {
		/* $ for the voxels on the two sides, the length shoulde be reconsidered $ */
		// if the voxel is on both the two sides (there is only one voxel in x dimension)
		if (d_cube_count[depth*3] == 1) {
			start_x = d_cube_start[d_max_depth[0]*3];
			len_x = d_cube_count[d_max_depth[0]*3];
		} else {
			if (node.cube_index[0] == 0) { // first cube in x dimension
				start_x = d_cube_start[d_max_depth[0]*3];
				len_x -= start_x % len_x;
			} else if (node.cube_index[0] == d_cube_count[depth*3] - 1) { // last in x dim
				len_x = (d_cube_start[d_max_depth[0]*3] + 
					d_cube_count[d_max_depth[0]*3] - 1) % len_x + 1;
			} 
		}

		/* if the voxel is on both the two sides (there is only on voxel in y dimension) */
		if (d_cube_count[depth*3+1] == 1) {
			start_y = d_cube_start[d_max_depth[0]*3+1];
			len_y = d_cube_count[d_max_depth[0]*3+1];
		} else {
			if (node.cube_index[1] == 0) { // first cube in y dimension
				start_y = d_cube_start[d_max_depth[0]*3+1];
				len_y -= start_y % len_y;
			} else if (node.cube_index[1] == d_cube_count[depth*3+1] - 1) { // last in y dim 
				len_y = (d_cube_start[d_max_depth[0]*3+1] + 
					d_cube_count[d_max_depth[0]*3+1] - 1) % len_y + 1;
			} 
		}

		/* if the voxel is on both two sides (there is only on voxel in z dimension) */
		if (d_cube_count[depth*3+2] == 1) {
			start_z = d_cube_start[d_max_depth[0]*3+2];
			len_z = d_cube_count[d_max_depth[0]*3+2];
		} else {
			if (node.cube_index[2] == 0) { // first cube in z dimension
				start_z = d_cube_start[d_max_depth[0]*3+2];
				len_z -= start_z % len_z;
			} else if (node.cube_index[2] == d_cube_count[depth*3+2] - 1) { // last in z dim 
				len_z = (d_cube_start[d_max_depth[0]*3+2] + 
					d_cube_count[d_max_depth[0]*3+2] - 1) % len_z + 1;
			} 
		}
	}
}

__device__
void genTetra(
	const OctFace &face, const char *vol_data, Tetra *tetra, char &tetra_count
){
	OctNode node1, node2;

	if (face.index1 != INVALID_NODE)
		node1 = dev_octlvl_ptr[face.level1][face.index1];
	if (face.index2 != INVALID_NODE)
		node2 = dev_octlvl_ptr[face.level2][face.index2];
	
	char side; // side of the count cube
	unsigned short len_x, len_y, len_z;       // length of voxels in the finest depth
	unsigned short start_x, start_y, start_z; // start of voxel offsets in the finest depth

	tetra_count = 0;

	// level is greater means the node is in the lower level with a smaller cube
	if (face.index1 != INVALID_NODE && face.level1 >= face.level2) {
		side = 'l'; // l means the coordinate in the specific dimension of the node is less
		getNodeLenStart(
			node1, face.level1, start_x, start_y, start_z, len_x, len_y, len_z);
	} else if (face.index2 != INVALID_NODE) {
		side = 'r'; // r means the coordinate in the specific dimension of the node is greater
		getNodeLenStart(
			node2, face.level2, start_x, start_y, start_z, len_x, len_y, len_z);
	} else
		return;

	XYZ p[4];
	float w[4];
	unsigned short x, y, z;
	unsigned short end_x = start_x+len_x, end_y = start_y+len_y, end_z = start_z+len_z;

	switch(face.face_dir) {
	case XY_PLANE:
		if (side == 'r') {
			end_z = start_z;
		} else {
			start_z += len_z;
			end_z = start_z;
		}
		break;
	case XZ_PLANE:
		if (side == 'r') {
			end_y = start_y;
		} else {
			start_y += len_y;
			end_y = start_y;
		}
		break;
	case YZ_PLANE:
		if (side == 'r') {
			end_x = start_x;
		} else {
			start_x += len_x;
			end_x = start_x;
		}
		break;
	}


	/////////////////////////////////////////////////////////////////
	// for debug !!
	//if (face.level1 == 8 && face.index1 == 419521 && face.level2 == 7 && face.index2 == 105292) {
	//	d_dbg_buf[0][0] = start_x;
	//	d_dbg_buf[0][1] = start_y;
	//	d_dbg_buf[0][2] = start_z;
	//	d_dbg_buf[0][3] = len_x;
	//	d_dbg_buf[0][4] = len_y;
	//	d_dbg_buf[0][5] = len_z;
	//	d_dbg_buf[0][6] = end_x;
	//	d_dbg_buf[0][7] = end_y;
	//	d_dbg_buf[0][8] = end_z;
	//}
	//////////////////////////////////////////////////////////////////


	char i = 0;
	for (x = start_x; x <= end_x; x += len_x) {
		for (y = start_y; y <= end_y; y += len_y) {
			for (z = start_z; z <= end_z; z += len_z) {
				// note here!! the offset of voxel data must subtract the start coordinate
				getVoxelData(vol_data, w[i], 
					x-d_cube_start[d_max_depth[0]*3], 
					y-d_cube_start[d_max_depth[0]*3+1], 
					z-d_cube_start[d_max_depth[0]*3+2]);
				p[i].x = x*d_slice_thick[0];
				p[i].y = y*d_slice_thick[1];
				p[i].z = z*d_slice_thick[2];
				i ++;
			}
		}
	}

	if (face.face_dir == XY_PLANE || face.face_dir == YZ_PLANE) {
		if (face.index1 != INVALID_NODE) {
			tetra[tetra_count].p[0] = p[1];
			tetra[tetra_count].val[0] = w[1];
			tetra[tetra_count].p[1].x = node1.dual_vert.x;
			tetra[tetra_count].p[1].y = node1.dual_vert.y;
			tetra[tetra_count].p[1].z = node1.dual_vert.z;
			tetra[tetra_count].val[1] = node1.dual_vert.w;
			tetra[tetra_count].p[2] = p[2];
			tetra[tetra_count].val[2] = w[2];
			tetra[tetra_count].p[3] = p[0];
			tetra[tetra_count].val[3] = w[0];
			tetra_count ++;

			tetra[tetra_count].p[0] = p[1];
			tetra[tetra_count].val[0] = w[1];
			tetra[tetra_count].p[1].x = node1.dual_vert.x;
			tetra[tetra_count].p[1].y = node1.dual_vert.y;
			tetra[tetra_count].p[1].z = node1.dual_vert.z;
			tetra[tetra_count].val[1] = node1.dual_vert.w;
			tetra[tetra_count].p[2] = p[3];
			tetra[tetra_count].val[2] = w[3];
			tetra[tetra_count].p[3] = p[2];
			tetra[tetra_count].val[3] = w[2];
			tetra_count ++;
		}

		if (face.index2 != INVALID_NODE) {
			tetra[tetra_count].p[0].x = node2.dual_vert.x;
			tetra[tetra_count].p[0].y = node2.dual_vert.y;
			tetra[tetra_count].p[0].z = node2.dual_vert.z;
			tetra[tetra_count].val[0] = node2.dual_vert.w;
			tetra[tetra_count].p[1] = p[1];
			tetra[tetra_count].val[1] = w[1];
			tetra[tetra_count].p[2] = p[2];
			tetra[tetra_count].val[2] = w[2];
			tetra[tetra_count].p[3] = p[0];
			tetra[tetra_count].val[3] = w[0];
			tetra_count ++;

			tetra[tetra_count].p[0].x = node2.dual_vert.x;
			tetra[tetra_count].p[0].y = node2.dual_vert.y;
			tetra[tetra_count].p[0].z = node2.dual_vert.z;
			tetra[tetra_count].val[0] = node2.dual_vert.w;
			tetra[tetra_count].p[1] = p[1];
			tetra[tetra_count].val[1] = w[1];
			tetra[tetra_count].p[2] = p[3];
			tetra[tetra_count].val[2] = w[3];
			tetra[tetra_count].p[3] = p[2];
			tetra[tetra_count].val[3] = w[2];
			tetra_count ++;
		}
	} else {
		if (face.index1 != INVALID_NODE) {
			tetra[tetra_count].p[0].x = node1.dual_vert.x;
			tetra[tetra_count].p[0].y = node1.dual_vert.y;
			tetra[tetra_count].p[0].z = node1.dual_vert.z;
			tetra[tetra_count].val[0] = node1.dual_vert.w;
			tetra[tetra_count].p[1] = p[1];
			tetra[tetra_count].val[1] = w[1];
			tetra[tetra_count].p[2] = p[2];
			tetra[tetra_count].val[2] = w[2];
			tetra[tetra_count].p[3] = p[0];
			tetra[tetra_count].val[3] = w[0];
			tetra_count ++;

			tetra[tetra_count].p[0].x = node1.dual_vert.x;
			tetra[tetra_count].p[0].y = node1.dual_vert.y;
			tetra[tetra_count].p[0].z = node1.dual_vert.z;
			tetra[tetra_count].val[0] = node1.dual_vert.w;
			tetra[tetra_count].p[1] = p[1];
			tetra[tetra_count].val[1] = w[1];
			tetra[tetra_count].p[2] = p[3];
			tetra[tetra_count].val[2] = w[3];
			tetra[tetra_count].p[3] = p[2];
			tetra[tetra_count].val[3] = w[2];
			tetra_count ++;
		}

		if (face.index2 != INVALID_NODE) {
			tetra[tetra_count].p[0] = p[1];
			tetra[tetra_count].val[0] = w[1];
			tetra[tetra_count].p[1].x = node2.dual_vert.x;
			tetra[tetra_count].p[1].y = node2.dual_vert.y;
			tetra[tetra_count].p[1].z = node2.dual_vert.z;
			tetra[tetra_count].val[1] = node2.dual_vert.w;
			tetra[tetra_count].p[2] = p[2];
			tetra[tetra_count].val[2] = w[2];
			tetra[tetra_count].p[3] = p[0];
			tetra[tetra_count].val[3] = w[0];
			tetra_count ++;

			tetra[tetra_count].p[0] = p[1];
			tetra[tetra_count].val[0] = w[1];
			tetra[tetra_count].p[1].x = node2.dual_vert.x;
			tetra[tetra_count].p[1].y = node2.dual_vert.y;
			tetra[tetra_count].p[1].z = node2.dual_vert.z;
			tetra[tetra_count].val[1] = node2.dual_vert.w;
			tetra[tetra_count].p[2] = p[3];
			tetra[tetra_count].val[2] = w[3];
			tetra[tetra_count].p[3] = p[2];
			tetra[tetra_count].val[3] = w[2];
			tetra_count ++;
		}
	}
}

__global__
void getIsosurfCountKn(
	const char* vol_data, const int start_depth, const float isovalue, 
	unsigned int *tri_count 
){
	const unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;

	int depth;
	for (depth = start_depth; depth <= d_max_depth[0]; depth ++)
		if (tid < d_n_mface[depth])
			break;

	if (depth > d_max_depth[0])
		return;

	unsigned int fid = tid;
	if (depth > start_depth) {
		fid -= d_n_mface[depth-1];
	}

	OctFace face = d_mface_ptr[depth][fid];

	unsigned char n_tri = 0;

	Tetra tetra[4];
	char tetra_count;
	genTetra(face, vol_data, tetra, tetra_count);

	for (char i = 0; i < tetra_count; i ++) {
		n_tri += PolygoniseTriGetCount(tetra[i], isovalue);
	}

	tri_count[tid] = n_tri;
}

template<typename T>
__forceinline__ __device__ 
void copy9(T *dst, const T *src) {
	dst[0] = src[0];
	dst[1] = src[1];
	dst[2] = src[2];
	dst[3] = src[3];
	dst[4] = src[4];
	dst[5] = src[5];
	dst[6] = src[6];
	dst[7] = src[7];
	dst[8] = src[8];
}

// for debug
__forceinline__ __device__ 
void copyTetra(Tetra *t, unsigned int n, float *buf) {
	buf[0] = n;
	int i, j;
	for (i = 0; i < n; i ++) {
		for (j = 0; j < 4; j ++) {
			buf[1+i*16+j*4] = t[i].p[j].x;
			buf[1+i*16+j*4+1] = t[i].p[j].y;
			buf[1+i*16+j*4+2] = t[i].p[j].z;
			buf[1+i*16+j*4+3] = t[i].val[j];
		}
	}
}

__global__
void genIsosurfKn(
	const char *vol_data, const int start_depth, const float isovalue, 
	const unsigned int *tri_addr, float *tri 
){
	/*!!const*/ unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;

	//////////////////////////////////////
	// debug !!
	//if (tid > 0)
	//	return;
	//tid = 65400;
	//////////////////////////////////////

	int depth;
	for (depth = start_depth; depth <= d_max_depth[0]; depth ++)
		if (tid < d_n_mface[depth])
			break;

	if (depth > d_max_depth[0])
		return;

	unsigned int addr = 0;
	unsigned int n_tri = 0;
	// please ACTIVE the code after DEBUG !!
	if (tid > 0)
		addr = tri_addr[tid-1];
	n_tri = tri_addr[tid];

	if (n_tri - addr <= 0)
		return;

	unsigned int fid = tid;
	if (depth > start_depth) {
		fid -= d_n_mface[depth-1];
	}

	OctFace face = d_mface_ptr[depth][fid];

	Tetra tetra[4];
	char tetra_count;
	genTetra(face, vol_data, tetra, tetra_count);

	/////////////////////////////////////
	// debug !!
	//copyTetra(tetra, tetra_count, tri);
	//return;
	/////////////////////////////////////

	float ptri[18];
	n_tri = 0;

	char i, j;
	for (i = 0; i < tetra_count; i ++) {
		n_tri = PolygoniseTri(tetra[i], isovalue, ptri);
		for (j = 0; j < n_tri; j ++)
			copy9(tri+(addr+j)*9, ptri+j*9);
		addr += n_tri;
	}

	////////////////////////
	// for debug !!
	//tri[20] = addr;
}

#endif