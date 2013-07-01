/*
 *  Kernels of generating edges for the 
 *  adaptive octree in parallel
 *
 *  Author: Ht
 *  Email : waytofall916 at gmail dot com
 *
 *  Copyright (C) Ht. All rights reserved.
 */
 
#ifndef _OCTREE_EDGE_KERNEL_H_
#define _OCTREE_EDGE_KERNEL_H_
 
/*******************************************************
 
    the nodes in the edge here, their 
    relative position:
  
           dm3cnt1 dm3cnt2
 
          +-------+-------+
          |       |       |
          | node1 | node2 |  dim2count1
          | (0,0) | (0,1) |
    dim2  +-------+-------+
          |       |       |
          | node4 | node3 |  dim2count2
          | (1,0) | (1,1) |
          +-------+-------+
                 dim3
	
    for the X axis edge, dim1 is x, dim2 is y, dim3 is z
    for the Y axis edge, dim1 is y, dim2 is x, dim3 is z
    for the Z axis edge, dim1 is z, dim2 is x, dim3 is y
 
********************************************************/

// get edge split count from child config of a node in an edge
__forceinline__ __device__
void getChildconfEdgeSplitCount(
	unsigned char &count, const unsigned char &child_config, const unsigned char &dir
){
	if (child_config == 0) {
		count = 0;
		return;
	}

	switch (dir) {
	case X_AXIS:
		getXConfig(child_config, count);
		configToCount(count);
		break;
	case Y_AXIS:
		getYConfig(child_config, count);
		configToCount(count);
		break;
	case Z_AXIS:
		getZConfig(child_config, count);
		configToCount(count);
		break;
	}
}

// get the split count of the given edge
__forceinline__ __device__
void getEdgeSplitCount(
	const OctEdge &edge, unsigned char &count, const OctNode *level_ptr, const int &depth
){
	unsigned char child_config = 0;
	count = 0;

	if (edge.level1 == depth && edge.index1 != INVALID_NODE) {
		child_config = level_ptr[edge.index1].child_config;
		if (child_config != 0) {
			getChildconfEdgeSplitCount(count, child_config, edge.edge_dir);
			return;
		}
	}
	if (edge.level2 == depth && edge.index2 != INVALID_NODE) {
		child_config = level_ptr[edge.index2].child_config;
		if (child_config != 0) {
			getChildconfEdgeSplitCount(count, child_config, edge.edge_dir);
			return;
		}
	}
	if (edge.level3 == depth && edge.index3 != INVALID_NODE) {
		child_config = level_ptr[edge.index3].child_config;
		if (child_config != 0) {
			getChildconfEdgeSplitCount(count, child_config, edge.edge_dir);
			return;
		}
	}
	if (edge.level4 == depth && edge.index4 != INVALID_NODE) {
		child_config = level_ptr[edge.index4].child_config;
		if (child_config != 0) {
			getChildconfEdgeSplitCount(count, child_config, edge.edge_dir);
			return;
		}
	}
}

// kernel for retrieving the split count of each edge
__global__
void getEdgeSplitCountKn(
	const OctEdge *edge_arr, unsigned int *split_count_arr, const OctNode *level_ptr, 
	const int depth, const unsigned int edge_count
){
	const unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid >= edge_count)
		return;

	const OctEdge edge = edge_arr[tid];
	unsigned char split_count;
	getEdgeSplitCount(edge, split_count, level_ptr, depth);

	split_count_arr[tid] = split_count;
}

// get split count in dimension 1, 2, 3 of an edge from the child_config of a node in that edge
__forceinline__ __device__
void getDim123Count(
	const unsigned char &edge_dir, const unsigned char &child_config, 
	unsigned char &dim1_count, unsigned char &dim2_count, unsigned char &dim3_count
){
	switch (edge_dir) {
	case X_AXIS:
		getXConfig(child_config, dim1_count);
		configToCount(dim1_count);
		getYConfig(child_config, dim2_count);
		configToCount(dim2_count);
		getZConfig(child_config, dim3_count);
		configToCount(dim3_count);
		break;
	case Y_AXIS:
		getYConfig(child_config, dim1_count);
		configToCount(dim1_count);
		getXConfig(child_config, dim2_count);
		configToCount(dim2_count);
		getZConfig(child_config, dim3_count);
		configToCount(dim3_count);
		break;
	case Z_AXIS:
		getZConfig(child_config, dim1_count);
		configToCount(dim1_count);
		getXConfig(child_config, dim2_count);
		configToCount(dim2_count);
		getYConfig(child_config, dim3_count);
		configToCount(dim3_count);
		break;
	}
}

// kernel for retrieving the split count of each edge
__global__
void splitEdgeKn(
	const OctEdge *splitin_edge_arr, OctEdge *splited_edge_arr, 
	const unsigned int *splited_addr_arr, const OctNode *level_ptr, 
	const unsigned int *oct_lvl_child_addr, const int depth, 
	const unsigned int edge_count
){
	const unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid >= edge_count)
		return;

	const OctEdge edge = splitin_edge_arr[tid];
	unsigned int splited_addr = 0;
	// note here! the splitted edge address is stored one place ahead
	if (tid > 0)
		splited_addr = splited_addr_arr[tid-1];
	
	char child_config1=0, child_config2=0, child_config3=0, child_config4=0;
	unsigned int child_addr1, child_addr2, child_addr3, child_addr4;
	// for the X axis edge, dim1 is x, dim2 is y, dim3 is z
	// for the Y axis edge, dim1 is y, dim2 is x, dim3 is z
	// for the Z axis edge, dim1 is z, dim2 is x, dim3 is y
	// node1 and node2 forms dim2count1, node4 and node3 forms dim2count2,
	// node1 and node4 forms dim3count1, node2 and node3 forms dim3count2
	// 255 means the count is invalid
	unsigned char dim1_count=255, dim2_count1=255, dim2_count2=255, 
		dim3_count1=255, dim3_count2=255;
	
	if (edge.level1 == depth && edge.index1 != INVALID_NODE) {
		child_config1 = level_ptr[edge.index1].child_config;
		if (child_config1) {
			getDim123Count(edge.edge_dir, child_config1, dim1_count, 
							dim2_count1, dim3_count1);
		}
		
		child_addr1 = 0;
		// !! note here, the child adress is stored one place ahead
		if (edge.index1 > 0)
			child_addr1 = oct_lvl_child_addr[edge.index1-1];
	}
	if (edge.level2 == depth && edge.index2 != INVALID_NODE) {
		child_config2 = level_ptr[edge.index2].child_config;		
		if (child_config2) {
			getDim123Count(edge.edge_dir, child_config2, dim1_count, 
							dim2_count1, dim3_count2);
		}
		
		child_addr2 = 0;
		// !! note here, the child adress is stored one place ahead
		if (edge.index2 > 0)
			child_addr2 = oct_lvl_child_addr[edge.index2-1];
	}
	if (edge.level3 == depth && edge.index3 != INVALID_NODE) {
		child_config3 = level_ptr[edge.index3].child_config;
		if (child_config3) {
			getDim123Count(edge.edge_dir, child_config3, dim1_count, 
							dim2_count2, dim3_count2);
		}
		
		child_addr3 = 0;
		// !! note here, the child adress is stored one place ahead
		if (edge.index3 > 0)
			child_addr3 = oct_lvl_child_addr[edge.index3-1];
	}
	if (edge.level4 == depth && edge.index4 != INVALID_NODE) {
		child_config4 = level_ptr[edge.index4].child_config;
		if (child_config4) {
			getDim123Count(edge.edge_dir, child_config4, dim1_count, 
							dim2_count2, dim3_count1);
		}
		
		child_addr4 = 0;
		// !! note here, the child adress is stored one place ahead
		if (edge.index4 > 0)
			child_addr4 = oct_lvl_child_addr[edge.index4-1];
	}
	
	unsigned char child_index[3], coord_map[3] = {0, 1, 2}, true_index[3], true_count[3];
	
	switch(edge.edge_dir) {
	case Y_AXIS:
		coord_map[0] = 1;
		coord_map[1] = 0;
		coord_map[2] = 2;
		break;
	case Z_AXIS:
		coord_map[0] = 2;
		coord_map[1] = 0;
		coord_map[2] = 1;
		break;
	}
	
	OctEdge new_edge;
	new_edge.edge_dir = edge.edge_dir;
	unsigned char local_index;
	for (child_index[0] = 0; child_index[0] < dim1_count; child_index[0] ++, splited_addr ++) {
		if (child_config1 != 0) {
			new_edge.level1 = depth + 1;
			child_index[1] = dim2_count1-1;
			child_index[2] = dim3_count1-1;
			getOctChildLocalIndex(local_index, child_index, dim1_count, dim2_count1, 
				dim3_count1, true_index, true_count, coord_map);
			new_edge.index1 = child_addr1 + local_index;
		} else {
			new_edge.level1 = edge.level1;
			new_edge.index1 = edge.index1;
		}
		
		if (child_config2 != 0) {
			new_edge.level2 = depth + 1;
			child_index[1] = dim2_count1-1;
			child_index[2] = 0;
			getOctChildLocalIndex(local_index, child_index, dim1_count, dim2_count1, 
				dim3_count2, true_index, true_count, coord_map);
			new_edge.index2 = child_addr2 + local_index;
		} else {
			new_edge.level2 = edge.level2;
			new_edge.index2 = edge.index2;
		}
		
		if (child_config3 != 0) {
			new_edge.level3 = depth + 1;
			child_index[1] = 0;
			child_index[2] = 0;
			getOctChildLocalIndex(local_index, child_index, dim1_count, dim2_count2, 
				dim3_count2, true_index, true_count, coord_map);
			new_edge.index3 = child_addr3 + local_index;
		} else {
			new_edge.level3 = edge.level3;
			new_edge.index3 = edge.index3;
		}
		
		if (child_config4 != 0) {
			new_edge.level4 = depth + 1;
			child_index[1] = 0;
			child_index[2] = dim3_count1-1;
			getOctChildLocalIndex(local_index, child_index, dim1_count, dim2_count2, 
				dim3_count1, true_index, true_count, coord_map);
			new_edge.index4 = child_addr4 + local_index;
		} else {
			new_edge.level4 = edge.level4;
			new_edge.index4 = edge.index4;
		}
		
		splited_edge_arr[splited_addr] = new_edge;
	}
}

// kernel for retrieving the new edge count incurred by expanding nodes in one level
__global__
void getOctLevelNewEdgeCountKn(
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
		case 1:
			count = 0;
			break;
		case 2:
			count = 1;
			break;
		case 3:
			count = 6;
			break;
		}
	}

	new_count[tid] = count;
}

// get the new edges incurred in one axis by the node
__forceinline__ __device__
void getNodeAxisNewEdge(
	unsigned char &index1, unsigned char &index2, unsigned char &index3, 
	const unsigned char &count1, uchar3arr &index_arr, uchar3arr &count_arr, 
	const int &depth, pOctEdge &new_edge_arr, unsigned int &new_addr, OctEdge &edge,
	const unsigned int &child_addr
){
	unsigned char local_index;

	for (index1 = 0; index1 < count1; index1 ++, new_addr ++) {
		edge.level1 = depth+1;
		index2 = 0;
		index3 = 0;
		getOctChildLocalIndex(local_index, index_arr, count_arr);
		edge.index1 = child_addr+local_index;
		
		edge.level2 = depth+1;
		index2 = 0;
		index3 = 1;
		getOctChildLocalIndex(local_index, index_arr, count_arr);
		edge.index2 = child_addr+local_index;
		
		edge.level3 = depth+1;
		index2 = 1;
		index3 = 1;
		getOctChildLocalIndex(local_index, index_arr, count_arr);
		edge.index3 = child_addr+local_index;
		
		edge.level4 = depth+1;
		index2 = 1;
		index3 = 0;
		getOctChildLocalIndex(local_index, index_arr, count_arr);
		edge.index4 = child_addr+local_index;
		
		new_edge_arr[new_addr] = edge;
	}
}

// kernel for generating the new edges incurred by expanding the nodes in one level
__global__
void fillOctLvlNewEdgeKn(
	const OctNode *level_ptr, const unsigned int *child_addr_arr, 
	const unsigned int *new_addr_arr, const unsigned int level_count, const int depth,  
	OctEdge *new_edge_arr
){
	const unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid >= level_count)
		return;
	
	const unsigned int child_config = level_ptr[tid].child_config;
	if (child_config == 0)
		return;
	
	unsigned int child_addr = 0;
	unsigned int new_addr = 0;
	// !! note here, the child & new edge address is stored one place ahead
	if (tid > 0) {
		child_addr = child_addr_arr[tid-1];
		new_addr = new_addr_arr[tid-1];
	}
	
	unsigned char child_index[3], child_count[3];
	
	getXConfig(child_config, child_count[0]);
	configToCount(child_count[0]);
	getYConfig(child_config, child_count[1]);
	configToCount(child_count[1]);
	getZConfig(child_config, child_count[2]);
	configToCount(child_count[2]);
	
	OctEdge edge;
	
	if (child_count[1] == 2 && child_count[2] == 2) {
		edge.edge_dir = X_AXIS;
		getNodeAxisNewEdge(child_index[0], child_index[1], child_index[2], 
			child_count[0], child_index, child_count, depth, new_edge_arr, new_addr, 
			edge, child_addr);
	}
	if (child_count[0] == 2 && child_count[2] == 2) {
		edge.edge_dir = Y_AXIS;
		getNodeAxisNewEdge(child_index[1], child_index[0], child_index[2], 
			child_count[1], child_index, child_count, depth, new_edge_arr, new_addr, 
			edge, child_addr);
	}
	if (child_count[0] == 2 && child_count[1] == 2) {
		edge.edge_dir = Z_AXIS;
		getNodeAxisNewEdge(child_index[2], child_index[0], child_index[1], 
			child_count[2], child_index, child_count, depth, new_edge_arr, new_addr, 
			edge, child_addr);
	}
}

// kernel for retrieving the new edge count incurred by splitting faces in one level
__global__
void getFaceNewEdgeCountKn(
	const OctFace *face_arr, unsigned int *new_count_arr, const int depth, 
	const OctNode *level_ptr, const unsigned int face_count
){
	const unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid >= face_count)
		return;
	
	const OctFace face = face_arr[tid];
	char whole_count = 0, new_count = 0;
	unsigned char _child_config = 0;
	
	if (face.level1 == depth && face.index1 != INVALID_NODE) {
		_child_config = level_ptr[face.index1].child_config;
	}
	if (_child_config == 0 && face.level2 == depth && face.index2 != INVALID_NODE) {
		_child_config = level_ptr[face.index2].child_config;
	}

	if (_child_config) {
		unsigned char config;
		
		switch(face.face_dir) {
		case XY_PLANE:
			getXConfig(_child_config, config);
			if (config == WHOLE_OCCUPIED)
				whole_count ++;
			getYConfig(_child_config, config);
			if (config == WHOLE_OCCUPIED)
				whole_count ++;
			break;
		case XZ_PLANE:
			getXConfig(_child_config, config);
			if (config == WHOLE_OCCUPIED)
				whole_count ++;
			getZConfig(_child_config, config);
			if (config == WHOLE_OCCUPIED)
				whole_count ++;
			break;
		case YZ_PLANE:
			getYConfig(_child_config, config);
			if (config == WHOLE_OCCUPIED)
				whole_count ++;
			getZConfig(_child_config, config);
			if (config == WHOLE_OCCUPIED)
				whole_count ++;
			break;
		}

		switch(whole_count) {
		case 0:
			new_count = 0;
			break;
		case 1:
			new_count = 1;
			break;
		case 2:
			new_count = 4;
			break;
		}
	}
	
	new_count_arr[tid] = new_count;
}

// get the new edges incurred by face in which dim3 varies 
__forceinline__ __device__
void genFaceOneDimNewEdgeDim3Vari(
	OctEdge &edge, const int &depth,  const OctFace &face, 
	const unsigned char &child_config1, const unsigned char &child_config2, 
	unsigned int &new_addr, const unsigned int &child_addr1, const unsigned int &child_addr2, 
	unsigned char &index1, unsigned char &index2, unsigned char &index3, 
	const unsigned char &dim1_count, const unsigned char &dim2_count, unsigned char &dim3_count, 
	const unsigned char &dim3_count1, const unsigned char &dim3_count2, 
	uchar3arr &index_arr, uchar3arr &count_arr, pOctEdge &new_edge_arr 
){
	unsigned char local_index;
	
	for (index1 = 0; index1 < dim1_count; index1 ++, new_addr ++) {
		if (child_config1 != 0) {
			index2 = 0;
			index3 = dim3_count1-1;
			dim3_count = dim3_count1;
			getOctChildLocalIndex(local_index, index_arr, count_arr);
			edge.level1 = depth+1;
			edge.index1 = child_addr1+local_index;
		} else {
			edge.level1 = face.level1;
			edge.index1 = face.index1;
		}
		
		if (child_config2 != 0) {
			index2 = 0;
			index3 = 0;
			dim3_count = dim3_count2;
			getOctChildLocalIndex(local_index, index_arr, count_arr);
			edge.level2 = depth+1;
			edge.index2 = child_addr2+local_index;
		} else {
			edge.level2 = face.level2;
			edge.index2 = face.index2;
		}
		
		if (child_config2 != 0) {
			index2 = dim2_count-1;
			index3 = 0;
			dim3_count = dim3_count2;
			getOctChildLocalIndex(local_index, index_arr, count_arr);
			edge.level3 = depth+1;
			edge.index3 = child_addr2+local_index;
		} else {
			edge.level3 = face.level2;
			edge.index3 = face.index2;
		}
		
		if (child_config1 != 0) {
			index2 = dim2_count-1;
			index3 = dim3_count1-1;
			dim3_count = dim3_count1;
			getOctChildLocalIndex(local_index, index_arr, count_arr);
			edge.level4 = depth+1;
			edge.index4 = child_addr1+local_index;
		} else {
			edge.level4 = face.level1;
			edge.index4 = face.index1;
		}
		
		new_edge_arr[new_addr] = edge;
	}
}

// get the new edges incurred by face in which dim2 varies 
__forceinline__ __device__
void genFaceOneDimNewEdgeDim2Vari(
	OctEdge &edge, const int &depth, const OctFace &face, 
	const unsigned char &child_config1, const unsigned char &child_config2, 
	unsigned int &new_addr, const unsigned int &child_addr1, const unsigned int &child_addr2, 
	unsigned char &index1, unsigned char &index2, unsigned char &index3, 
	const unsigned char &dim1_count, unsigned char &dim2_count, const unsigned char &dim3_count, 
	const unsigned char &dim2_count1, const unsigned char &dim2_count2, 
	uchar3arr &index_arr, uchar3arr &count_arr, pOctEdge &new_edge_arr 
){
	unsigned char local_index;
	
	for (index1 = 0; index1 < dim1_count; index1 ++, new_addr ++) {
		if (child_config1 != 0) {
			index2 = dim2_count1-1;
			index3 = 0;
			dim2_count = dim2_count1;
			getOctChildLocalIndex(local_index, index_arr, count_arr);
			edge.level1 = depth+1;
			edge.index1 = child_addr1+local_index;
		} else {
			edge.level1 = face.level1;
			edge.index1 = face.index1;
		}
		
		if (child_config1 != 0) {
			index2 = dim2_count1-1;
			index3 = dim3_count-1;
			dim2_count = dim2_count1;
			getOctChildLocalIndex(local_index, index_arr, count_arr);
			edge.level2 = depth+1;
			edge.index2 = child_addr1+local_index;
		} else {
			edge.level2 = face.level1;
			edge.index2 = face.index1;
		}
		
		if (child_config2 != 0) {
			index2 = 0;
			index3 = dim3_count-1;
			dim2_count = dim2_count2;
			getOctChildLocalIndex(local_index, index_arr, count_arr);
			edge.level3 = depth+1;
			edge.index3 = child_addr2+local_index;
		} else {
			edge.level3 = face.level2;
			edge.index3 = face.index2;
		}
		
		if (child_config2 != 0) {
			index2 = 0;
			index3 = 0;
			dim2_count = dim2_count2;
			getOctChildLocalIndex(local_index, index_arr, count_arr);
			edge.level4 = depth+1;
			edge.index4 = child_addr2+local_index;
		} else {
			edge.level4 = face.level2;
			edge.index4 = face.index2;
		}
		
		new_edge_arr[new_addr] = edge;
#ifdef __CUDA_DBG
		//printEdge(edge);
#endif
	}
}

// kernel for generating new edges incurred by splitting faces in one level
__global__
void fillFaceNewEdgeKn(
	const OctFace *face_arr, const OctNode *level_ptr, 
	const unsigned int *child_addr_arr, const unsigned int *new_addr_arr, 
	const unsigned int face_count, const int depth, OctEdge *new_edge_arr
){
	const unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid >= face_count)
		return;

	//////////////////////////
	// for debug !!
	//if (tid != 316)
	//	return;
	//tid = 12135;
	
	const OctFace face = face_arr[tid];
	
	unsigned char child_config1 = 0, child_config2 = 0;
	
	if (face.level1 == depth && face.index1 != INVALID_NODE) {
		child_config1 = level_ptr[face.index1].child_config;
	}
	if (face.level2 == depth && face.index2 != INVALID_NODE) {
		child_config2 = level_ptr[face.index2].child_config;
	}

#ifdef __CUDA_DBG
	//if (tid == 316) {
		//CUPRINTF("child config1 %x, child config2 %x\n", child_config1, child_config2);
	//}
#endif
	
	if (child_config1 == 0 && child_config2 == 0)
		return;
		
	unsigned int child_addr1 = 0, child_addr2 = 0;
	if (face.level1 == depth && face.index1 != INVALID_NODE && face.index1 > 0) {
		// !! note here, the child adress is stored one place ahead
		child_addr1 = child_addr_arr[face.index1-1];
	}
	if (face.level2 == depth && face.index2 != INVALID_NODE && face.index2 > 0) {
		// !! note here, the child adress is stored one place ahead
		child_addr2 = child_addr_arr[face.index2-1];
	}
	
	unsigned int new_addr = 0;
	// !! note here, new edge address is stored one place ahead
	if (tid > 0) 
		new_addr = new_addr_arr[tid-1];

#ifdef __CUDA_DBG
	//CUPRINTF("new addr,  %u\n", new_addr);
	unsigned int addr_count = new_addr_arr[tid];
	addr_count -= new_addr;
	unsigned int write_count = new_addr;
#endif

	unsigned char child_index[3], child_count[3], left_dcount1, left_dcount2;
	
	OctEdge edge;
	switch(face.face_dir) {
		unsigned char child_config;
	case XY_PLANE:
		if (child_config1 != 0) {
			getZConfig(child_config1, left_dcount1);
			configToCount(left_dcount1);
		}
		if (child_config2 != 0) {
			getZConfig(child_config2, left_dcount2);
			configToCount(left_dcount2);
		}
			
		if (child_config1 != 0)
			child_config = child_config1;
		else
			child_config = child_config2;
			
		getXConfig(child_config, child_count[0]);
		configToCount(child_count[0]);
		getYConfig(child_config, child_count[1]);
		configToCount(child_count[1]);
		
		// X_AXIS new edge in XY_PLANE face
		if (child_count[1] == 2) {
			edge.edge_dir = X_AXIS;
			genFaceOneDimNewEdgeDim3Vari(
				edge, depth, face, 
				child_config1, child_config2, 
				new_addr, child_addr1, child_addr2, 
				child_index[0], child_index[1], child_index[2], 
				child_count[0], child_count[1], child_count[2], 
				left_dcount1, left_dcount2, 
				child_index, child_count, new_edge_arr);
		}
		
		// Y_AXIS new edge in XY_PLANE face
		if (child_count[0] == 2) {
			edge.edge_dir = Y_AXIS;
			genFaceOneDimNewEdgeDim3Vari(
				edge, depth, face, 
				child_config1, child_config2, 
				new_addr, child_addr1, child_addr2, 
				child_index[1], child_index[0], child_index[2], 
				child_count[1], child_count[0], child_count[2], 
				left_dcount1, left_dcount2, 
				child_index, child_count, new_edge_arr);
		}
		break;
	case XZ_PLANE:
		if (child_config1 != 0) {
			getYConfig(child_config1, left_dcount1);
			configToCount(left_dcount1);
		}
		if (child_config2 != 0) {
			getYConfig(child_config2, left_dcount2);
			configToCount(left_dcount2);
		}
			
		if (child_config1 != 0)
			child_config = child_config1;
		else
			child_config = child_config2;
			
		getXConfig(child_config, child_count[0]);
		configToCount(child_count[0]);
		getZConfig(child_config, child_count[2]);
		configToCount(child_count[2]);
			
		// X_AXIS new edge in XZ_PLANE face
		if (child_count[2] == 2) {
			edge.edge_dir = X_AXIS;
			genFaceOneDimNewEdgeDim2Vari(
				edge, depth, face, 
				child_config1, child_config2, 
				new_addr, child_addr1, child_addr2, 
				child_index[0], child_index[1], child_index[2], 
				child_count[0], child_count[1], child_count[2], 
				left_dcount1, left_dcount2, 
				child_index, child_count, new_edge_arr);
		}
			
		// Z_AXIS new edge in XZ_PLANE face
		if (child_count[0] == 2) {
			edge.edge_dir = Z_AXIS;
			genFaceOneDimNewEdgeDim3Vari(
				edge, depth, face, 
				child_config1, child_config2, 
				new_addr, child_addr1, child_addr2, 
				child_index[2], child_index[0], child_index[1], 
				child_count[2], child_count[0], child_count[1], 
				left_dcount1, left_dcount2, 
				child_index, child_count, new_edge_arr);
		}
		break;
	case YZ_PLANE:
		if (child_config1 != 0) {
			getXConfig(child_config1, left_dcount1);
			configToCount(left_dcount1);
		}
		if (child_config2 != 0) {
			getXConfig(child_config2, left_dcount2);
			configToCount(left_dcount2);
		}
			
		if (child_config1 != 0)
			child_config = child_config1;
		else
			child_config = child_config2;
			
		getYConfig(child_config, child_count[1]);
		configToCount(child_count[1]);
		getZConfig(child_config, child_count[2]);
		configToCount(child_count[2]);
			
		// Y_AXIS new edge in YZ_PLANE face
		if (child_count[2] == 2) {
			edge.edge_dir = Y_AXIS;
			genFaceOneDimNewEdgeDim2Vari(
				edge, depth, face, 
				child_config1, child_config2, 
				new_addr, child_addr1, child_addr2, 
				child_index[1], child_index[0], child_index[2], 
				child_count[1], child_count[0], child_count[2], 
				left_dcount1, left_dcount2, 
				child_index, child_count, new_edge_arr);
		}
		
		// Z_AXIS new edge in YZ_PLANE face
		if (child_count[1] == 2) {
			edge.edge_dir = Z_AXIS;
			genFaceOneDimNewEdgeDim2Vari(
				edge, depth, face, 
				child_config1, child_config2, 
				new_addr, child_addr1, child_addr2, 
				child_index[2], child_index[0], child_index[1], 
				child_count[2], child_count[0], child_count[1], 
				left_dcount1, left_dcount2, 
				child_index, child_count, new_edge_arr);
		}
		break;
	}

#ifdef __CUDA_DBG
	//if (tid == 316) {
		write_count = new_addr - write_count;
		if (write_count != addr_count) {
			CUPRINTF("ERROR!! tid %u, addr count %u, write count %u\n", tid, addr_count, write_count);
			printFace(face);
		}
	//}
#endif
}

// kernel for generating the first level edges
__global__ 
void makeFirstLevelEdgeKn (OctEdge *edge_arr, const int depth)
{
	unsigned short x_dim = d_cube_count[depth*3];
	unsigned short y_dim = d_cube_count[depth*3+1];
	unsigned short z_dim = d_cube_count[depth*3+2];

	// count of edges parallel to each axis
	__shared__ unsigned int edge_count_x, edge_count_y, edge_count_z;
	if (threadIdx.x == 0) {
		edge_count_x = x_dim* (y_dim+1)* (z_dim+1);
		edge_count_y = (x_dim+1)* y_dim* (z_dim+1);
		edge_count_z = (x_dim+1)* (y_dim+1)* z_dim;
	}
	__syncthreads();

	unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid >= edge_count_x + edge_count_y + edge_count_z)
		return;
	
	OctEdge edge;
	edge.level1 = edge.level2 = edge.level3 = edge.level4 = depth;
	edge.index1 = edge.index2 = edge.index3 = edge.index4 = INVALID_NODE;
	
	short x, y, z;
	// parallel to x axis
	if (tid < edge_count_x) { 
		edge.edge_dir = X_AXIS;

		getCoordinate(tid, x, y, z, x_dim, y_dim+1);
		// in node index, z is the last computed dimension
		if (y > 0 && z > 0)
			getLinearId(edge.index1, x, y-1, z-1, x_dim, y_dim);
		if (y > 0 && z < z_dim)
			getLinearId(edge.index2, x, y-1, z,   x_dim, y_dim);
		if (y < y_dim && z > 0)
			getLinearId(edge.index4, x, y,   z-1, x_dim, y_dim);
		if (y < y_dim && z < z_dim)
			getLinearId(edge.index3, x, y,   z,   x_dim, y_dim);
	} 
	// parallel to y axis
	else if (tid < edge_count_x + edge_count_y) { 
		edge.edge_dir = Y_AXIS;

		tid -= edge_count_x;
		getCoordinate(tid, x, y, z, x_dim+1, y_dim);
		tid += edge_count_x;
		
		if (x > 0 && z > 0)
			getLinearId(edge.index1, x-1, y, z-1, x_dim, y_dim);
		if (x > 0 && z < z_dim)
			getLinearId(edge.index2, x-1, y, z,   x_dim, y_dim);
		if (x < x_dim && z > 0)
			getLinearId(edge.index4, x,   y, z-1, x_dim, y_dim);
		if (x < x_dim && z < z_dim)
			getLinearId(edge.index3, x,   y, z,   x_dim, y_dim);
	}
	// parallel to z axis
	else { 
		edge.edge_dir = Z_AXIS;

		tid -= (edge_count_x + edge_count_y);
		getCoordinate(tid, x, y, z, x_dim+1, y_dim+1);
		tid += (edge_count_x + edge_count_y);
		
		if (x > 0 && y > 0)
			getLinearId(edge.index1, x-1, y-1, z, x_dim, y_dim);
		if (x > 0 && y < y_dim)
			getLinearId(edge.index2, x-1, y,   z, x_dim, y_dim);
		if (x < x_dim && y > 0)
			getLinearId(edge.index4, x,   y-1, z, x_dim, y_dim);
		if (x < x_dim && y < y_dim)
			getLinearId(edge.index3, x,   y,   z, x_dim, y_dim);
	}
	
	//printEdge(edge);
	edge_arr[tid] = edge;
}

#endif // _OCTREE_EDGE_KERNEL_H_