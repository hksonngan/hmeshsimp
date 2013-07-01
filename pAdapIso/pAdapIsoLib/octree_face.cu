/*
 *  Kernels of generating faces for the 
 *  adaptive octree in parallel
 *
 *  Author: Ht
 *  Email : waytofall916 at gmail dot com
 *
 *  Copyright (C) Ht. All rights reserved.
 */

#ifndef _OCTREE_FACE_KERNEL_H_
#define _OCTREE_FACE_KERNEL_H_

/************************************************************

    ________________________
   |\           \           \
   | \           \           \
   |  \___________\___________\
   |  |           |           |
   |  |           |           |
   \  |    node1  |    node2  |
    \ |           |           |
	 \|___________|___________|


   the face is formed between two nodes, the coordinate of 
   two dimensions are the same and that of one dimension 
   differs, the coordinate of node1 in the "diff" dimension 
   is less than that of nodes

*************************************************************/

__forceinline__ __device__
void getCoordinate(
	const unsigned int &id, short &x, short &y, short &z, 
	const unsigned short x_dim, const unsigned short y_dim
){
	x = id % x_dim;
	y = id % (x_dim * y_dim) / x_dim;
	z = id / (x_dim * y_dim);
}

__forceinline__ __device__
void getLinearId(
	unsigned int &id, const short &x, const short &y, const short &z, 
	const unsigned short &x_dim, const unsigned short &y_dim
){
	id = x + y*x_dim + z*x_dim*y_dim;
}

// kernel for generating the first level faces
__global__
void makeFirstLevelFaceKn(OctFace *face_ptr, const int depth) 
{
	unsigned short x_dim = d_cube_count[depth*3];
	unsigned short y_dim = d_cube_count[depth*3+1];
	unsigned short z_dim = d_cube_count[depth*3+2];

	// face count parallel to different coordinate planes
	__shared__ unsigned int face_count_xy, face_count_xz, face_count_yz;
	if (threadIdx.x == 0) {
		face_count_xy = x_dim* y_dim* (z_dim+1);
		face_count_xz = x_dim* (y_dim+1)* z_dim;
		face_count_yz = (x_dim+1)* y_dim* z_dim;
	}
	__syncthreads();

	unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid >= face_count_xy + face_count_xz + face_count_yz)
		return;

	OctFace face;
	face.level1 = depth;
	face.level2 = depth;
	face.index1 = INVALID_NODE;
	face.index2 = INVALID_NODE;
	face.split_addr = 0;

	short x, y, z;
	// parallel to xy coordinate plane
	if (tid < face_count_xy) {
		face.face_dir = XY_PLANE;

		getCoordinate(tid, x, y, z, x_dim, y_dim);
		if (z > 0)
			getLinearId(face.index1, x, y, z-1, x_dim, y_dim);
		if (z < z_dim)
			getLinearId(face.index2, x, y, z, x_dim, y_dim);
	}
	// parallel to xz coordinate plane 
	else if (tid < face_count_xy + face_count_xz) {
		face.face_dir = XZ_PLANE;

		tid -= face_count_xy;
		getCoordinate(tid, x, y, z, x_dim, y_dim+1);
		tid += face_count_xy;

		if (y > 0)
			getLinearId(face.index1, x, y-1, z, x_dim, y_dim);
		if (y < y_dim)
			getLinearId(face.index2, x, y, z, x_dim, y_dim);
	}
	// parallel to yz coordinate plane
	else {
		face.face_dir = YZ_PLANE;

		tid -= (face_count_xy + face_count_xz);
		getCoordinate(tid, x, y, z, x_dim+1, y_dim);
		tid += (face_count_xy + face_count_xz);

		if (x > 0)
			getLinearId(face.index1, x-1, y, z, x_dim, y_dim);
		if (x < x_dim)
			getLinearId(face.index2, x, y, z, x_dim, y_dim);
	}

	face_ptr[tid] = face;
}

// get the split count for a face from the child_config of a node in that face
__forceinline__ __device__
void getSplitCount(
	unsigned int &count, const unsigned char &child_config, const unsigned char &dir
){
	if (child_config == 0) {
		count = 0;
		return;
	}

	unsigned char config;
	switch (dir) {
	case XY_PLANE:
		getXConfig(child_config, config);
		configToCount(config);
		count = config;
		getYConfig(child_config, config);
		configToCount(config);
		count *= config;
		break;
	case XZ_PLANE:
		getXConfig(child_config, config);
		configToCount(config);
		count = config;
		getZConfig(child_config, config);
		configToCount(config);
		count *= config;
		break;
	case YZ_PLANE:
		getYConfig(child_config, config);
		configToCount(config);
		count = config;
		getZConfig(child_config, config);
		configToCount(config);
		count *= config;
		break;
	}
}

// get the split count for given face 
__forceinline__ __device__
void getFaceSplitCount(OctFace &face, const OctNode *level_ptr, const int &depth) {
	unsigned char child_config1 = 0;
	unsigned char child_config2 = 0;

	if (face.level1 == depth && face.index1 != INVALID_NODE) {
		child_config1 = level_ptr[face.index1].child_config;
	}
	if (face.level2 == depth && face.index2 != INVALID_NODE) {
		child_config2 = level_ptr[face.index2].child_config;
	}

	if (child_config1 != 0) {
		getSplitCount(face.split_addr, child_config1, face.face_dir);
	} else if (child_config2 != 0) {
		getSplitCount(face.split_addr, child_config2, face.face_dir);
	} else {
		face.split_addr = 0;
	}
}

// kernel for retrieving split count for faces in one level
__global__
void getFaceSplitCountKn(
	OctFace *face_ptr, const OctNode *level_ptr, const int depth, const unsigned int face_count
){
	const unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid >= face_count)
		return;

	OctFace face = face_ptr[tid];
	getFaceSplitCount(face, level_ptr, depth);

	face_ptr[tid].split_addr = face.split_addr;
}

// get the local index of a child in an octree node
__forceinline__ __device__
void getOctChildLocalIndex(
	unsigned char &local_index, const uchar3arr &index, const unsigned char &dim1_count, 
	const unsigned char &dim2_count, const unsigned char &dim3_count, 
	uchar3arr &true_index, uchar3arr &true_count, const uchar3arr &map
){
	// reorder the index and count the x, y, z sequence
	true_index[map[0]] = index[0];
	true_index[map[1]] = index[1];
	true_index[map[2]] = index[2];

	true_count[map[0]] = dim1_count;
	true_count[map[1]] = dim2_count;
	true_count[map[2]] = dim3_count;

	// true_index and true_count is now the sequence of x, y, z
	// note that in the kernel 'makeOctLevelChildKn' the local
	// child index is computed such that x is the outer-most loop
	// and than y and z is the inner-most loop
	local_index = true_index[0]* true_count[1]* true_count[2] + 
		true_index[1]* true_count[2] + true_index[2];
}

// kernel for generating the splitted faces for faces in one level
__global__
void splitFaceKn(
	const OctFace *splitin_face_ptr, OctFace *splited_face_ptr, const OctNode *level_ptr, 
	unsigned int *oct_lvl_child_addr, const int depth, const unsigned int face_count
){
	const unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid >= face_count)
		return;

	OctFace face = splitin_face_ptr[tid];
	
	OctFace splited_face;
	splited_face.face_dir = face.face_dir;
	splited_face.split_addr = 0;
	// for debug !!
	//splited_face.split_addr = tid;

	// octree node child config of node1, node2 
	unsigned char child_config1 = 0, child_config2 = 0, child_config;
	// octree node child start address of node1, node2
	unsigned int child_addr1, child_addr2; 

	child_addr1 = face.split_addr;
	getFaceSplitCount(face, level_ptr, depth);
	if (face.split_addr == 0)
		return;

	// the split_addr is the inclusive sum, subtract the count of its own
	face.split_addr = child_addr1 - face.split_addr;

	if (face.level1 == depth && face.index1 != INVALID_NODE) {
		child_config1 = level_ptr[face.index1].child_config;
		child_addr1 = 0;
		// !! note here, the child adress is stored one place ahead
		if (face.index1 > 0)
			child_addr1 = oct_lvl_child_addr[face.index1-1];
	}
	if (face.level2 == depth && face.index2 != INVALID_NODE) {
		child_config2 = level_ptr[face.index2].child_config;
		child_addr2 = 0;
		if (face.index2 > 0)
			child_addr2 = oct_lvl_child_addr[face.index2-1];
	}

	if (child_config1 != 0)
		child_config = child_config1;
	else 
		child_config = child_config2;

	// map that can make the reordered sequenced trasferred back to the orignial sequece
	unsigned char coord_map[3][3] = {
		{0, 1, 2},
		{0, 2, 1},
		{1, 2, 0} };
	unsigned char child_index[3]/*reordered local index of the child in the node*/, 
		child_count[3], config, true_index[3], true_count[3], dim3_count1, dim3_count2;

	// reorder the index sequence, get the count of split in each dimension
	// for the count in the dimension on the plane parallel to the splitting
	// plane, the count and coordinate equals, while in the other dimension
	// the count and coordinate differs
	switch(face.face_dir) {
	case XY_PLANE:
		getXConfig(child_config, config);
		getChildCount(config, child_count[0]);
		getYConfig(child_config, config);
		getChildCount(config, child_count[1]);
		getZConfig(child_config1, config);
		getChildCount(config, dim3_count1);
		getZConfig(child_config2, config);
		getChildCount(config, dim3_count2);
		break;
	case XZ_PLANE:
		getXConfig(child_config, config);
		getChildCount(config, child_count[0]);
		getZConfig(child_config, config);
		getChildCount(config, child_count[1]);
		getYConfig(child_config1, config);
		getChildCount(config, dim3_count1);
		getYConfig(child_config2, config);
		getChildCount(config, dim3_count2);
		break;
	case YZ_PLANE:
		getYConfig(child_config, config);
		getChildCount(config, child_count[0]);
		getZConfig(child_config, config);
		getChildCount(config, child_count[1]);
		getXConfig(child_config1, config);
		getChildCount(config, dim3_count1);
		getXConfig(child_config2, config);
		getChildCount(config, dim3_count2);
		break;
	}

	// use the same register
	unsigned char &local_index = child_config;
	unsigned char &i = config;

	i = 0;
	for (child_index[0] = 0; child_index[0] < child_count[0]; child_index[0] ++) {
		for (child_index[1] = 0; child_index[1] < child_count[1]; child_index[1] ++) {
			if (child_config1 != 0) {
				child_index[2] = dim3_count1-1;
				child_count[2] = dim3_count1;
				getOctChildLocalIndex(local_index, child_index, child_count[0], 
					child_count[1], child_count[2], true_index, true_count, 
					coord_map[face.face_dir]);
				splited_face.level1 = depth + 1;
				splited_face.index1 = child_addr1 + local_index;
			} else {
				splited_face.level1 = face.level1;
				splited_face.index1 = face.index1;
			}

			if (child_config2 != 0) {
				child_index[2] = 0;
				child_count[2] = dim3_count2;
				getOctChildLocalIndex(local_index, child_index, child_count[0], 
					child_count[1], child_count[2], true_index, true_count, 
					coord_map[face.face_dir]);
				splited_face.level2 = depth + 1;
				splited_face.index2 = child_addr2 + local_index;
			} else {
				splited_face.level2 = face.level2;
				splited_face.index2 = face.index2;
			}

			splited_face_ptr[face.split_addr + i] = splited_face;
			i ++;
		}
	}
}

// kernel for retrieving new face count incurred by expanding nodes in one level
__global__
void getOctLevelNewFaceCountKn(
	const OctNode *level_ptr, unsigned int *new_count, const unsigned int level_count
){
	const unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid >= level_count)
		return;

	const unsigned char child_config = level_ptr[tid].child_config;
	unsigned int count;

	if (child_config == 0) {
		count = 0;
	} else {
		unsigned char config;
		char whole_count = 0;

		getXConfig(child_config, config);
		if (config == WHOLE_OCCUPIED)
			whole_count ++;
		getYConfig(child_config, config);
		if (config == WHOLE_OCCUPIED)
			whole_count ++;
		getZConfig(child_config, config);
		if (config == WHOLE_OCCUPIED)
			whole_count ++;

		switch(whole_count) {
		case 0:
			count = 0;
			break;
		case 1:
			count = 1;
			break;
		case 2:
			count = 4;
			break;
		case 3:
			count = 12;
			break;
		}
	}

	new_count[tid] = count;
}

__forceinline__ __device__
void getOctChildLocalIndex(
	unsigned char &local_index, const uchar3arr &index, const uchar3arr &count
){
	local_index = index[0]* count[1]* count[2] + index[1]* count[2] + index[2];
}

// kernel for generating new faces incurred by expanding nodes in one level
__global__
void fillNewFaceKn(
	const OctNode *level_ptr, const unsigned int *oct_child_addr, const unsigned int *new_addr, 
	OctFace *face_ptr, const unsigned int level_count, const int depth
){
	const unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid >= level_count)
		return;

	const unsigned char child_config = level_ptr[tid].child_config;
	if (child_config == 0)
		return;

	unsigned int new_face_addr = 0;
	unsigned int child_start_addr = 0;
	if (tid > 0) {
		new_face_addr = new_addr[tid-1];
		child_start_addr = oct_child_addr[tid-1];
	}

	OctFace new_face;
	new_face.level1 = new_face.level2 = depth;
	// for debug !!
	//new_face.split_addr = tid;

	unsigned char config, child_index[3]/*x, y, z*/, child_count[3];

	getXConfig(child_config, config);
	getChildCount(config, child_count[0]);

	getYConfig(child_config, config);
	getChildCount(config, child_count[1]);

	getZConfig(child_config, config);
	getChildCount(config, child_count[2]);

	unsigned char &local_index = config;

	new_face.face_dir = XY_PLANE;
	if (child_count[2] == 2) {
		for (child_index[0] = 0; child_index[0] < child_count[0]; child_index[0] ++) {
			for (child_index[1] = 0; child_index[1] < child_count[1]; child_index[1] ++) {
				child_index[2] =  0;
				getOctChildLocalIndex(local_index, child_index, child_count);
				new_face.index1 = child_start_addr + local_index;

				child_index[2] =  1;
				getOctChildLocalIndex(local_index, child_index, child_count);
				new_face.index2 = child_start_addr + local_index;

				face_ptr[new_face_addr] = new_face;
				new_face_addr ++;
			}
		}
	}

	new_face.face_dir = XZ_PLANE;
	if (child_count[1] == 2) {
		for (child_index[0] = 0; child_index[0] < child_count[0]; child_index[0] ++) {
			for (child_index[2] = 0; child_index[2] < child_count[2]; child_index[2] ++) {
				child_index[1] =  0;
				getOctChildLocalIndex(local_index, child_index, child_count);
				new_face.index1 = child_start_addr + local_index;

				child_index[1] =  1;
				getOctChildLocalIndex(local_index, child_index, child_count);
				new_face.index2 = child_start_addr + local_index;

				face_ptr[new_face_addr] = new_face;
				new_face_addr ++;
			}
		}
	}

	new_face.face_dir = YZ_PLANE;
	if (child_count[0] == 2) {
		for (child_index[1] = 0; child_index[1] < child_count[1]; child_index[1] ++) {
			for (child_index[2] = 0; child_index[2] < child_count[2]; child_index[2] ++) {
				child_index[0] =  0;
				getOctChildLocalIndex(local_index, child_index, child_count);
				new_face.index1 = child_start_addr + local_index;

				child_index[0] =  1;
				getOctChildLocalIndex(local_index, child_index, child_count);
				new_face.index2 = child_start_addr + local_index;

				face_ptr[new_face_addr] = new_face;
				new_face_addr ++;
			}
		}
	}
}

#endif