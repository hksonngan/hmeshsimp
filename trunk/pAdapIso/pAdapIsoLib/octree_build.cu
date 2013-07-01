/*
 *  Kernels for building the adaptive octree in parallel
 *
 *  Author: Ht
 *  Email : waytofall916 at gmail dot com
 *
 *  Copyright (C) Ht. All rights reserved.
 */


/*******************************************************************
  NOTE:
  There are two kinds of COORDINATE for the CUBE (or POINTS on the 
  cube), the ABSOLUTE coordinate and the OFFSET coordinate. The
  abs coord is the coordinate in the 4-dimension and its usage is
  for the ease of compute of the coordinate and count in different 
  levels of the octree, while the offset coord is just the offset
  in the 3-dimensional volume data.
*******************************************************************/

#ifndef _OCTREE_KERNEL_H_
#define _OCTREE_KERNEL_H_

#include "iso_math.cu"
#include "child_config.cu"

#define CUBE_SAMPLE_COUNT 10 // count of sampled points in a cube/voxel

enum CU_DATA_TYPE {
	CHAR_TYPE    = 0,
	UCHAR_TYPE   = 1,
	SHORT_TYPE   = 2,
	USHORT_TYPE  = 3,
	FLOAT_TYPE   = 4
};

template<typename T> 
__forceinline__ __device__ 
void getVoxelData(
    const char *vol_data, T &rvalue, const unsigned short &x, 
	const unsigned short &y, const unsigned short &z
){
	unsigned int offset = 
		z* (d_cube_count[d_max_depth[0]*3]+1)* (d_cube_count[d_max_depth[0]*3+1]+1)
		+ y* (d_cube_count[d_max_depth[0]*3]+1) + x;

	switch (d_data_format[0]) {
	case CHAR_TYPE:
		rvalue = *((char*)(vol_data+ offset* sizeof(char)));
		break;
	case UCHAR_TYPE:
		rvalue = *((unsigned char*)(vol_data+ offset* sizeof(unsigned char)));
		break;
	case SHORT_TYPE:
		rvalue = *((short*)(vol_data+ offset* sizeof(short)));
		break;
	case USHORT_TYPE:
		rvalue = *((unsigned short*)(vol_data+ offset* sizeof(unsigned short)));
		break;
	}
}

template<typename T> 
__forceinline__ __device__ 
void normalize(T &a, T &b, T &c, T &d) {
	T l = a*a + b*b + c*c + d*d;
	l = sqrtf(l);
	a /= l;
	b /= l;
	c /= l;
	d /= l;
}

// get the hyperplane parameters for the sample point given
// all the coordinate here is the absolute coordinate in the finest depth
// i've mad bug here and all the count and offset (coordinate) thing here
// is tricky!
template<typename T>
__forceinline__ __device__
void evaluateHyperplane(
	const unsigned short &abs_x, const unsigned short &abs_y, const unsigned short &abs_z, 
	const char *vol_data, T &a, T &b, T &c, T &d, T &e, T &w
){
	// offset of each dimension in the 3-dim array
	const unsigned short off_x = abs_x - d_cube_start[d_max_depth[0]*3];
	const unsigned short off_y = abs_y - d_cube_start[d_max_depth[0]*3+1];
	const unsigned short off_z = abs_z - d_cube_start[d_max_depth[0]*3+2];

	//if (off_x < 0 || off_x > d_cube_count[d_max_depth[0]*3])
	//	return;
	//if (off_y < 0 || off_y > d_cube_count[d_max_depth[0]*3+1])
	//	return;
	//if (off_z < 0 || off_z > d_cube_count[d_max_depth[0]*3+2])
	//	return;

	// left most point
	if (off_x == 0)
		getVoxelData(vol_data, d, off_x, off_y, off_z);
	else
		getVoxelData(vol_data, d, off_x - 1, off_y, off_z);

	// right most point
	if (off_x == d_cube_count[d_max_depth[0]*3])
		getVoxelData(vol_data, a, off_x, off_y, off_z);
	else 
		getVoxelData(vol_data, a, off_x + 1, off_y, off_z);

	// get the x dimension partial derivative
	a -= d;
	a /= (2 * d_slice_thick[0]);

	// front most point
	if (off_y == 0)
		getVoxelData(vol_data, d, off_x, off_y, off_z);
	else
		getVoxelData(vol_data, d, off_x, off_y - 1, off_z);

	// back most point
	if (off_y == d_cube_count[d_max_depth[0]*3+1])
		getVoxelData(vol_data, b, off_x, off_y, off_z);
	else 
		getVoxelData(vol_data, b, off_x, off_y + 1, off_z);

	// get the y dimension partial derivative
	b -= d;
	b /= (2 * d_slice_thick[1]);

	// up most point
	if (off_z == 0)
		getVoxelData(vol_data, d, off_x, off_y, off_z);
	else
		getVoxelData(vol_data, d, off_x, off_y, off_z - 1);

	// down most point
	if (off_z == d_cube_count[d_max_depth[0]*3+2])
		getVoxelData(vol_data, c, off_x, off_y, off_z);
	else 
		getVoxelData(vol_data, c, off_x, off_y, off_z + 1);

	// get the y dimension partial derivative
	c -= d;
	c /= (2 * d_slice_thick[2]);

	d = -1;
	getVoxelData(vol_data, w, off_x, off_y, off_z);

	// I use the absolute coordinate here since when calculating
	// the formula using the point on the hyperplane, the point
	// in the 4-dimensinonal space is actually the 'absolute 
	// cooridinate' multiply the thickness
	normalize(a, b, c, d);
	e = - (a*abs_x*d_slice_thick[0] + b*abs_y*d_slice_thick[1] + 
		c*abs_z*d_slice_thick[2] + d*w);
}

// get the child config, dual vertex and error incurred for the given node
// only need to make the node index ready in variable 'node'
__forceinline__ __device__
void evaluateNode(
	OctNode &node, const int &depth, const char *vol_data, 
	const float &error_thresh, unsigned char &child_count
){
	/* ================== $ compute the length and child count for each node $ ================= */

	unsigned short len_x, len_y, len_z;       // length of voxels in the finest depth
	// length in each dimension equals to the cube size if it's not on the side
	len_x = d_cube_size[depth];
	len_y = len_z = len_x;
	// note the start here is in fact the coordinate in the finest depth
	unsigned short start_x, start_y, start_z; // start of voxel offsets in the finest depth
	// not for the starting one
	start_x = (node.cube_index[0] + d_cube_start[depth * 3]    ) * len_x;
	start_y = (node.cube_index[1] + d_cube_start[depth * 3 + 1]) * len_y;
	start_z = (node.cube_index[2] + d_cube_start[depth * 3 + 2]) * len_z;

	node.child_config = 0;

	if (depth != d_max_depth[0]) {
		/* $ for the voxels on the two sides, the length shoulde be reconsidered $ */
		// if the voxel is on both the two sides (there is only one voxel in x dimension)
		if (d_cube_count[depth * 3] == 1) {
			if (d_cube_start[d_max_depth[0] * 3] + 
				d_cube_count[d_max_depth[0] * 3] - 1 < start_x + len_x/2) {
				childConfigSetXLeft(node.child_config);
				child_count = 1;
			} else if (d_cube_start[d_max_depth[0] * 3] >= start_x + len_x/2) {
				childConfigSetXRight(node.child_config);
				child_count = 1;
			} else {
				childConfigSetXWhole(node.child_config);
				child_count = 2;
			}
			start_x = d_cube_start[d_max_depth[0] * 3];
			len_x = d_cube_count[d_max_depth[0] * 3];
		} else {
			// first cube in x dimension
			if (node.cube_index[0] == 0) { 
				if (d_cube_start[d_max_depth[0] * 3] >= start_x + len_x/2) {
					childConfigSetXRight(node.child_config);
					child_count = 1;
				} else { 
					childConfigSetXWhole(node.child_config);
					child_count = 2;
				}
				start_x = d_cube_start[d_max_depth[0] * 3];
				len_x -= start_x % len_x;
			} 
			// last in x dim
			else if (node.cube_index[0] == d_cube_count[depth * 3] - 1) { 
				if (d_cube_start[d_max_depth[0] * 3] + 
					d_cube_count[d_max_depth[0] * 3] - 1 < start_x + len_x/2) {
					childConfigSetXLeft(node.child_config);
					child_count = 1;
				} else { 
					childConfigSetXWhole(node.child_config);
					child_count = 2;
				}
				len_x = (d_cube_start[d_max_depth[0] * 3] + 
					d_cube_count[d_max_depth[0] * 3] - 1) % len_x + 1;
			} 
			else {
				childConfigSetXWhole(node.child_config);
				child_count = 2;
			}
		}

		/* if the voxel is on both the two sides (there is only on voxel in y dimension) */
		if (d_cube_count[depth * 3 + 1] == 1) {
			if (d_cube_start[d_max_depth[0] * 3 + 1] + 
				d_cube_count[d_max_depth[0] * 3 + 1] - 1 < start_y + len_y/2) {
				childConfigSetYLeft(node.child_config);
				child_count *= 1;
			} else if (d_cube_start[d_max_depth[0] * 3 + 1] >= start_y + len_y/2) {
				childConfigSetYRight(node.child_config);
				child_count *= 1;
			} else {
				childConfigSetYWhole(node.child_config);
				child_count *= 2;
			}
			start_y = d_cube_start[d_max_depth[0] * 3 + 1];
			len_y = d_cube_count[d_max_depth[0] * 3 + 1];
		} else {
			// first cube in y dimension
			if (node.cube_index[1] == 0) { 
				if (d_cube_start[d_max_depth[0] * 3 + 1] >= start_y + len_y/2) {
					childConfigSetYRight(node.child_config);
					child_count *= 1;
				} else {
					childConfigSetYWhole(node.child_config);
					child_count *= 2;
				}
				start_y = d_cube_start[d_max_depth[0] * 3 + 1];
				len_y -= start_y % len_y;
			} 
			// last in y dim 
			else if (node.cube_index[1] == d_cube_count[depth * 3 + 1] - 1) { 
				if (d_cube_start[d_max_depth[0] * 3 + 1] + 
					d_cube_count[d_max_depth[0] * 3 + 1] - 1 < start_y + len_y/2) {
					childConfigSetYLeft(node.child_config);
					child_count *= 1;
				} else {
					childConfigSetYWhole(node.child_config);
					child_count *= 2;
				}
				len_y = (d_cube_start[d_max_depth[0] * 3 + 1] + 
					d_cube_count[d_max_depth[0] * 3 + 1] - 1) % len_y + 1;
			} 
			else {
				childConfigSetYWhole(node.child_config);
				child_count *= 2;
			}
		}

		/* if the voxel is on both two sides (there is only on voxel in z dimension) */
		if (d_cube_count[depth * 3 + 2] == 1) {
			if (d_cube_start[d_max_depth[0] * 3 + 2] + 
				d_cube_count[d_max_depth[0] * 3 + 2] - 1 < start_z + len_z/2) {
				childConfigSetZLeft(node.child_config);
				child_count *= 1;
			}
			if (d_cube_start[d_max_depth[0] * 3 + 2] >= start_z + len_z/2) {
				childConfigSetZRight(node.child_config);
				child_count *= 1;
			} else {
				childConfigSetZWhole(node.child_config);
				child_count *= 2;
			}
			start_z = d_cube_start[d_max_depth[0] * 3 + 2];
			len_z = d_cube_count[d_max_depth[0] * 3 + 2];
		} else {
			// first cube in z dimension
			if (node.cube_index[2] == 0) { 
				if (d_cube_start[d_max_depth[0] * 3 + 2] >= start_z + len_z/2) {
					childConfigSetZRight(node.child_config);
					child_count *= 1;
				} else {
					childConfigSetZWhole(node.child_config);
					child_count *= 2;
				}
				start_z = d_cube_start[d_max_depth[0] * 3 + 2];
				len_z -= start_z % len_z;
			} 
			// last in z dim 
			else if (node.cube_index[2] == d_cube_count[depth * 3 + 2] - 1) { 
				if (d_cube_start[d_max_depth[0] * 3 + 2] + 
					d_cube_count[d_max_depth[0] * 3 + 2] - 1 < start_z + len_z/2) {
					childConfigSetZLeft(node.child_config);
					child_count *= 1;
				} else {
					childConfigSetZWhole(node.child_config);
					child_count *= 2;
				}
				len_z = (d_cube_start[d_max_depth[0] * 3 + 2] + 
					d_cube_count[d_max_depth[0] * 3 + 2] - 1) % len_z + 1;
			} 
			else {
				childConfigSetZWhole(node.child_config);
				child_count *= 2;
			}
		}

	}

	/* ================ $ compute the dual vertex and error incurred $ =============== */

	// evaluate every vertex and the hyperplane, then add it to the quadrice error matrix
	QeMatrix5<float> qem;
	float &a = node.dual_vert.x, &b = node.dual_vert.y, 
		&c = node.dual_vert.z, &d = node.dual_vert.w;
	float e, w, w_mean = 0.0f;
	char n = 0;
	short step_x = (len_x / 2 == 0 ? 1 : len_x / 2);
	short step_y = (len_y / 2 == 0 ? 1 : len_y / 2);
	short step_z = (len_z / 2 == 0 ? 1 : len_z / 2);

	// !! for debug
	//if (node.cube_index[0] == 50 && node.cube_index[1] == 25 && node.cube_index[2] == 6) {
	//	d_dbg_buf[0][0] = start_x;
	//	d_dbg_buf[0][1] = start_y;
	//	d_dbg_buf[0][2] = start_z;
	//	d_dbg_buf[0][3] = len_x;
	//	d_dbg_buf[0][4] = len_y;
	//	d_dbg_buf[0][5] = len_z;
	//	d_dbg_buf[0][6] = step_x;
	//	d_dbg_buf[0][7] = step_y;
	//	d_dbg_buf[0][8] = step_z;
	//}

	// for debug !!
	//int index = 0;

	short i, j, k;
	for (i = 0; i <= len_x; i += step_x) {
		for (j = 0; j <= len_y; j += step_y) {
			for (k = 0; k <= len_z; k += step_z) {
				evaluateHyperplane(
					start_x+i, start_y+j, start_z+k, vol_data, a, b, c, d, e, w);
				w_mean = w_mean*n/(n+1) + w/(n+1);
				n ++;
				qem.addPlane(a, b, c, d, e);

				// for debug !!
				//if (node.cube_index[0] == 50 && node.cube_index[1] == 25 && node.cube_index[2] == 6) {
				//	getVoxelData(vol_data, w, start_x+i-d_cube_start[d_max_depth[0]*3], 
				//		start_y+j-d_cube_start[d_max_depth[0]*3+1], start_z+k-d_cube_start[d_max_depth[0]*3]+2);
				//	d_dbg_buf[0][index] = w;
				//	index ++;
				//}
			}
		}
	}

	// try to solve the quadric matrix
	const float BOUND_DIV = 1000.0f;
	MatrixSolver5<float> solver;
	bool qemSolCanUse = true;
	solver.assign(qem);
	if (solver.solve(node.dual_vert.x, node.dual_vert.y, 
		node.dual_vert.z, node.dual_vert.w)) {
		// if the qem solution is out of boundary
		if (node.dual_vert.x < ((float)start_x+(float)len_x/BOUND_DIV)*d_slice_thick[0] || 
			node.dual_vert.x > ((float)start_x+(float)len_x-(float)len_x/BOUND_DIV)*d_slice_thick[0] ||
			node.dual_vert.y < ((float)start_y+(float)len_y/BOUND_DIV)*d_slice_thick[1] || 
			node.dual_vert.y > ((float)start_y+(float)len_y-(float)len_y/BOUND_DIV)*d_slice_thick[1] ||
			node.dual_vert.z < ((float)start_z+(float)len_z/BOUND_DIV)*d_slice_thick[2] || 
			node.dual_vert.z > ((float)start_z+(float)len_z-(float)len_z/BOUND_DIV)*d_slice_thick[2]
			) {
			qemSolCanUse = false;
		}
	} else {
		qemSolCanUse = false;
	}

	// if the linear system can't be sovled or the solution is out of 
	// the boundary, try the mean value
	if (!qemSolCanUse) {
		node.dual_vert.x = (start_x + ((float)len_x)/2) * d_slice_thick[0];
		node.dual_vert.y = (start_y + ((float)len_y)/2) * d_slice_thick[1];
		node.dual_vert.z = (start_z + ((float)len_z)/2) * d_slice_thick[2];
		node.dual_vert.w = w_mean;
	}

	if (depth != d_max_depth[0]) {
		float error;
		qem.evaluateError(node.dual_vert.x, node.dual_vert.y, 
			node.dual_vert.z, node.dual_vert.w, error);
		if (error <= error_thresh) {
			child_count = 0;
			// child_config equals to 0 means there is no child
			node.child_config = 0;
		}

		// for debug !!
		//node.dual_vert.x = error;
	}
}

// kernel for filling data into first octree level
__global__ void makeFirstOctLevelKn( 
	OctNode* level_ptr, unsigned int *child_addr, const int depth, 
	const char* vol_data, const float error_thresh 
){
	OctNode node;
	unsigned char child_count;
	const unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;

	node.cube_index[0] = tid % d_cube_count[depth * 3];
	node.cube_index[1] = tid % (d_cube_count[depth * 3] * d_cube_count[depth * 3 + 1])
							/ d_cube_count[depth * 3];
	node.cube_index[2] = tid / (d_cube_count[depth * 3] * d_cube_count[depth * 3 + 1]);

	if (node.cube_index[2] >= d_cube_count[depth * 3 + 2])
		return;

	evaluateNode(node, depth, vol_data, error_thresh, child_count);

	level_ptr[tid] = node;
	child_addr[tid] = child_count;
}

// kernel for filling data into the children of nodes in one level
__global__ void makeOctLevelChildKn(
	OctNode* level_ptr, unsigned int *child_addr, OctNode* child_level_ptr, 
	unsigned int *child_childaddr, const unsigned int level_count, 
	const int depth /* depth of the level making child, not the child's */, 
	const char* vol_data, const float error_thresh
){
	const unsigned int tid = threadIdx.x + blockIdx.x*blockDim.x;
	if (tid >= level_count)
		return;

	OctNode node = level_ptr[tid], child;
	if (node.child_config == 0)
		return;

	unsigned int child_start_addr = 0;
	if (tid > 0)
		child_start_addr = child_addr[tid-1];
	unsigned char child_count;

	unsigned char x_config, y_config, z_config;
	char i, j, k, child_index;
	bool x_continue, y_continue, z_continue;

	getXConfig(node.child_config, x_config);
	getYConfig(node.child_config, y_config);
	getZConfig(node.child_config, z_config);

	x_continue = true;
	child_index = 0;
	for (i = 0; x_continue; i ++) {
		switch (x_config) {
		case LEFT_OCCUPIED:
			child.cube_index[0] = 
				(node.cube_index[0]+d_cube_start[depth*3])*2 - d_cube_start[(depth+1)*3];
			x_continue = false;
			break;
		case RIGHT_OCCUPIED:
			child.cube_index[0] = 
				(node.cube_index[0]+d_cube_start[depth*3])*2+1 - d_cube_start[(depth+1)*3];
			x_continue = false;
			break;
		case WHOLE_OCCUPIED:
			child.cube_index[0] = 
				(node.cube_index[0]+d_cube_start[depth*3])*2+i - d_cube_start[(depth+1)*3];
			if (i >= 1)
				x_continue = false;
			break;
		default:
			x_continue = false;
		}

		y_continue = true; 
		for (j = 0; y_continue; j ++) {
			switch (y_config) {
			case LEFT_OCCUPIED:
				child.cube_index[1] = 
					(node.cube_index[1]+d_cube_start[depth*3+1])*2 - d_cube_start[(depth+1)*3+1];
				y_continue = false;
				break;
			case RIGHT_OCCUPIED:
				child.cube_index[1] = 
					(node.cube_index[1]+d_cube_start[depth*3+1])*2+1 - d_cube_start[(depth+1)*3+1];
				y_continue = false;
				break;
			case WHOLE_OCCUPIED:
				child.cube_index[1] = 
					(node.cube_index[1]+d_cube_start[depth*3+1])*2+j - d_cube_start[(depth+1)*3+1];
				if (j >= 1)
					y_continue = false;
				break;
			default:
				y_continue = false;
			}

			z_continue = true;
			for (k = 0; z_continue; k ++) {
				switch (z_config) {
				case LEFT_OCCUPIED:
					child.cube_index[2] = 
						(node.cube_index[2]+d_cube_start[depth*3+2])*2 - d_cube_start[(depth+1)*3+2];
					z_continue = false;
					break;
				case RIGHT_OCCUPIED:
					child.cube_index[2] = 
						(node.cube_index[2]+d_cube_start[depth*3+2])*2+1 - d_cube_start[(depth+1)*3+2];
					z_continue = false;
					break;
				case WHOLE_OCCUPIED:
					child.cube_index[2] = 
						(node.cube_index[2]+d_cube_start[depth*3+2])*2+k - d_cube_start[(depth+1)*3+2];
					if (k >= 1)
						z_continue = false;
					break;
				default:
					z_continue = false;
				}

				evaluateNode(child, depth+1, vol_data, error_thresh, child_count);

				child_level_ptr[child_start_addr+child_index] = child;
				if (depth+1 < d_max_depth[0])
					child_childaddr[child_start_addr+child_index] = child_count;
				child_index ++;
			}
		}
	}
}

//// kernel for filling data into one level of the octree
//__global__ void makeOctLevelKn(
//	OctNode* level_ptr, unsigned int *child_addr, const unsigned int level_count,
//	const int depth, const char* vol_data, const float error_thresh
//){
//	unsigned char child_count;
//	const unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
//	if (tid >= level_count)
//		return;
//
//	OctNode node = level_ptr[tid];
//	evaluateNode(node, depth, vol_data, error_thresh, child_count);
//
//	level_ptr[tid] = node;
//	child_addr[tid] = child_count;
//}

#endif // #ifndef _OCTREE_KERNEL_H_
