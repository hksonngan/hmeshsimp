/*
 *  Adaptively Generate the Iso-surfaces in Parallel
 *  Run On the Host Invoking the Cuda Kernel
 *
 *  Author: Ht
 *  Email : waytofall916 at gmail dot com
 *
 *  Copyright (C) Ht. All rights reserved.
 */

// projcet includes
#include "nvtimer.h"
#include "vol_set.h"
#include "p_adap_iso.h"
#include "tri_soup_stream.h"

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <climits>
#include <algorithm>

using std::cout;
using std::string;
using std::ostringstream;
using std::endl;

#define __VERBOSE
//#define __GPU_WRITE_BACK
//#define __CUDA_DBG

// includes CUDA
#include <cuda_runtime.h>
#include <thrust/scan.h>
#include <thrust/partition.h>
#include <thrust/iterator/zip_iterator.h>

template<typename Type>
inline Type MIN(Type a, Type b) { return a < b ? a : b; }

template<typename Type>
inline Type MAX(Type a, Type b) { return a > b ? a : b; }

// cuda files
#include "cuda_helper.cu"
#include "cuPrintf.cu"

//The macro CUPRINTF is defined for architectures
//with different compute capabilities.
#if __CUDA_ARCH__ < 200     //Compute capability 1.x architectures
#define CUPRINTF cuPrintf
#else                       //Compute capability 2.x architectures
#define CUPRINTF(fmt, ...) printf("[%d, %d]:\t" fmt, \
                                  blockIdx.y*gridDim.x+blockIdx.x,\
                                  threadIdx.z*blockDim.x*blockDim.y+threadIdx.y*blockDim.x+threadIdx.x,\
                                  __VA_ARGS__)
#endif

const unsigned int MAX_DEPTH_COUNT = 11;
const unsigned int MIN_GRID_COUNT = 4;


/*======================$ DEVICE SIDE DEFINITION $========================*/

typedef struct _XYZW { 
	float x, y, z, w; 
} XYZW;

// octree kernel type decalaration
typedef struct _OctNode {
	unsigned short cube_index[3];            // cube index in the specific depth
	unsigned char child_config;              // configuration of child
	XYZW dual_vert;  // dual vertex
} OctNode;

typedef OctNode *pOctNode;

enum FACE_DIRECTION {
	XY_PLANE = 0,
	XZ_PLANE = 1,
	YZ_PLANE = 2
};

enum EDGE_DIRECTION {
	X_AXIS = 0,
	Y_AXIS = 1,
	Z_AXIS = 2
};

typedef struct _OctFace {
	unsigned char face_dir;
	// left face
	char level1; // oct node1's level
	unsigned int index1; // oct node1 index in the level
	// right face
	char level2; // oct node2's level
	unsigned int index2; // oct node2 index int the level
	unsigned int split_addr;
} OctFace;

typedef struct _OctEdge {
	unsigned char edge_dir;
	char level1, level2, level3, level4; // adjacent oct nodes' level
	unsigned int index1, index2, index3, index4; // oct nodes' indices in the level
	//unsigned int split_addr;
} OctEdge;

typedef struct _Tetra {
	XYZ p[4];
	float val[4];
} Tetra;

typedef Tetra* pTetra;
typedef OctFace* pOctFace;
typedef OctEdge* pOctEdge;
typedef char* pchar;
typedef unsigned char uchar3arr[3];

#define INVALID_NODE UINT_MAX

// global cuda variables decalaration
__constant__ unsigned short
    d_cube_count[MAX_DEPTH_COUNT * 3], // a two-dimension array, the 2nd dimension is the
                                       // cube count in x, y, z dimension of the volume set
    d_cube_start[MAX_DEPTH_COUNT * 3], // a two-dimension array, the 2nd dimension is the cube's
                                       // start coordinate in x, y, z dimension of the volume set
    d_cube_size[MAX_DEPTH_COUNT];      // size of the cube in each level
__constant__ unsigned int d_data_format[1], d_max_depth[1];
__constant__ float d_slice_thick[3];

__constant__ OctNode* dev_octlvl_ptr[MAX_DEPTH_COUNT];
__constant__ unsigned int d_octlvl_count[MAX_DEPTH_COUNT];

__constant__ OctFace* d_mface_ptr[MAX_DEPTH_COUNT];
__constant__ unsigned int d_n_mface[MAX_DEPTH_COUNT];

__constant__ OctEdge* d_medge_ptr[MAX_DEPTH_COUNT];
__constant__ unsigned int d_n_medge[MAX_DEPTH_COUNT];

/*====================$ DEVICE SIDE DEFINITION {end} $======================*/

/*======================$ GLOBAL VARIABLES $========================*/

#ifdef __CUDA_DBG
float *h_dbg_buf_devptr;
const unsigned int dbg_buf_size = 1000;
__constant__ float *d_dbg_buf[1];
float hdbg_buf[dbg_buf_size];
#endif

// kernels
#include "dbg_help.cu"
#include "octree_build.cu"
#include "octree_face.cu"
#include "octree_edge.cu"
#include "iso_gen.cu"
#include "edge_iso_gen.cu"

// global variables for cpp
unsigned short maxVolLen, maxLenPow2;
unsigned int maxDepth;
unsigned short cubeStart[MAX_DEPTH_COUNT * 3], cubeCount[MAX_DEPTH_COUNT * 3],
	volSize[3], cubeSize[MAX_DEPTH_COUNT];
OctNode* d_octLvlPtr[MAX_DEPTH_COUNT];
unsigned int
	/*child start address in the next level for each node in a specific level*/
	*d_childAddrPtr[MAX_DEPTH_COUNT],
	/*count of nodes in each level*/
	level_count[MAX_DEPTH_COUNT];
char* d_volData;
unsigned int dataFormat;
VolumeSet volSet;

// pointers of minimla faces in each level
OctFace *h_mface_devptr[MAX_DEPTH_COUNT];
// count of minimla faces in each level
unsigned int n_mface[MAX_DEPTH_COUNT];

// pointers of minimla edges in each level
OctEdge *h_medge_devptr[MAX_DEPTH_COUNT];
// count of minimla edges in each level
unsigned int n_medge[MAX_DEPTH_COUNT];

unsigned int tri_count;
float *d_tri, *h_tri;

// for debug
OctNode *h_octLvlPtr[MAX_DEPTH_COUNT];
unsigned int *h_child_addr[MAX_DEPTH_COUNT];
OctFace *h_mface_ptr[MAX_DEPTH_COUNT];
OctEdge *h_medge_ptr[MAX_DEPTH_COUNT];

#include "padap_dbg.h"

bool use_cuPrintf;

/*======================$ GLOBAL VARIABLES {end} $========================*/

unsigned int getCubeCount(unsigned int lowerLayerStart, unsigned int lowerLayerCount) {
	// code here is a little tricky, however, it is
	// similar to the method in function getGridDim()
	if (lowerLayerStart % 2 == 0)
		return (lowerLayerCount + 1) / 2;
	else
		return lowerLayerCount / 2 + 1;
}

unsigned int getGridDim(unsigned int validThreadCount, unsigned int blockDim) {
	return (validThreadCount + blockDim - 1) / blockDim;
}

void getLinearBlockGridDim3(
	unsigned int validThreadCount, unsigned int linearBlockDim, dim3 &blockDim, dim3 &gridDim
){
	blockDim.y = blockDim.z = 1;
	gridDim.y = gridDim.z = 1;
	blockDim.x = linearBlockDim;
	gridDim.x = getGridDim(validThreadCount, blockDim.x);
}

// deprecated !!
void getFirstBlockGridDim(unsigned short *cubeCount, dim3 &blockDim, dim3 &gridDim) {
    // try 8x8x4
    blockDim.x = blockDim.y = blockDim.z = 8;
    if (cubeCount[0] <= MIN(cubeCount[1], cubeCount[2]))
        blockDim.x = 4;
    if (cubeCount[1] <= MIN(cubeCount[0], cubeCount[2]))
        blockDim.y = 4;
    else
        blockDim.z = 4;

    gridDim.x = getGridDim(cubeCount[0], blockDim.x);
    gridDim.y = getGridDim(cubeCount[1], blockDim.y);
    gridDim.z = getGridDim(cubeCount[2], blockDim.z);

    if (gridDim.x * gridDim.y * gridDim.z >= MIN_GRID_COUNT)
        return;

    // try 8x4x4
    blockDim.x = blockDim.y = blockDim.z = 4;
    if (cubeCount[0] >= MAX(cubeCount[1], cubeCount[2]))
        blockDim.x = 8;
    if (cubeCount[1] >= MAX(cubeCount[0], cubeCount[2]))
        blockDim.y = 8;
    else
        blockDim.z = 8;

    gridDim.x = getGridDim(cubeCount[0], blockDim.x);
    gridDim.y = getGridDim(cubeCount[1], blockDim.y);
    gridDim.z = getGridDim(cubeCount[2], blockDim.z);

    if (gridDim.x * gridDim.y * gridDim.z >= MIN_GRID_COUNT)
        return;

    // then it's 4x4x4
    blockDim.x = blockDim.y = blockDim.z = 4;
    gridDim.x = getGridDim(cubeCount[0], blockDim.x);
    gridDim.y = getGridDim(cubeCount[1], blockDim.y);
    gridDim.z = getGridDim(cubeCount[2], blockDim.z);
}

void trisToFile(const string& volfilename) {
	string filename = getFilename(volfilename.c_str());
	filename += ".tris";

	TriSoupStream tri_stream;

	float max_x, min_x, max_y, min_y, max_z, min_z;
	min_x = cubeStart[maxDepth*3] *volSet.thickness.s[0];
	max_x = (cubeStart[maxDepth*3] +cubeCount[maxDepth*3]) *volSet.thickness.s[0];
	min_y = cubeStart[maxDepth*3+1] *volSet.thickness.s[1];
	max_y = (cubeStart[maxDepth*3+1] +cubeCount[maxDepth*3+1]) *volSet.thickness.s[1];
	min_z = cubeStart[maxDepth*3+2] *volSet.thickness.s[2];
	max_z = (cubeStart[maxDepth*3+2] +cubeCount[maxDepth*3+2]) *volSet.thickness.s[2];

	tri_stream.setBoundBox(max_x, min_x, max_y, min_y, max_z, min_z);

	if (!tri_stream.openForWrite(filename.c_str())) {
		throw "error open tris file for write";
	}

	for (int i = 0; i < tri_count; i ++) {
		float *ptri = h_tri + i*9;
		tri_stream.writeFloat
			(ptri[0], ptri[1], ptri[2], ptri[3], ptri[4],
			ptri[5], ptri[6], ptri[7], ptri[8]);
	}

	tri_stream.closeForWrite();
}

void toRawTris(const string& volfilename, float *tri, int count, ostringstream &oss) {
	string filename = getFilename(volfilename.c_str());
	filename += ".rawtris";

	std::ofstream fout(filename, std::ios::binary | std::ios::out);
	if (!fout.good()) {
		throw "error open .rawtris file for write";
	}

	fout.write(reinterpret_cast<char*>(tri), count*9*sizeof(float));

	float min_x, max_x, min_y, max_y, min_z, max_z;
	int min_x_index=0, max_x_index=0, min_y_index=0, max_y_index=0, 
		min_z_index=0, max_z_index=0;

	min_x = max_x = tri[0];
	min_y = max_y = tri[1];
	min_z = max_z = tri[2];
	for (int i = 0; i < tri_count*3; i ++) {
		if (tri[i*3] > max_x) {
			max_x = tri[i*3];
			max_x_index = i;
		} else if (tri[i*3] < min_x) {
			min_x = tri[i*3];
			min_x_index = i;
		}
		if (tri[i*3+1] > max_y) {
			max_y = tri[i*3+1];
			max_y_index = i;
		} else if (tri[i*3+1] < min_y) {
			min_y = tri[i*3+1];
			min_y_index = i;
		}
		if (tri[i*3+2] > max_z) {
			max_z = tri[i*3+2];
			max_z_index = i;
		} else if (tri[i*3+2] < min_z) {
			min_z = tri[i*3+2];
			min_z_index = i;
		}
	}

	oss << endl << "isosurf bound box: " << endl
		<< "x: " << min_x << " " << max_x << endl
		<< "y: " << min_y << " " << max_y << endl
		<< "z: " << min_z << " " << max_z << endl
		<< "x, min index: " << min_x_index << " max index: " << max_x_index << endl
		<< "y, min index: " << min_y_index << " max index: " << max_y_index << endl
		<< "z, min index: " << min_z_index << " max index: " << max_z_index << endl;
}

void to_Tris(float *tri, int count, char *suffix = NULL) {
	std::string filename = getFilename(volSet.dataFileName.c_str());
	if (suffix)
		filename += suffix;
	filename += "._tris";
	std::ofstream fout(filename);

	for (int i = 0; i < count; i ++) {
		for (int j = 0; j < 9; j ++)
			fout << tri[i*9+j] << std::endl;
	}
}

void writeCuprintf() {
	if (use_cuPrintf) {
		FILE *dbg_fout = fopen ("dbg_out.txt","w");
		fprintf(dbg_fout, "\n\n");

		//Dump current contents of output buffer to standard
		//output, and origin (block id and thread id) of each line
		//of output is enabled(true).
		cudaPrintfDisplay(dbg_fout, true);
	}
}

void buildOctree(const int startDepth, const float errorThresh, ___OUT string& info)
{
	const unsigned int MAX_BLOCK_DIM = 128;
	const unsigned int MIN_BLOCK_DIM = 64;

	/* == compute the cube size, depth, start cordinate for the octree in each depth == */

	int i;

	volSize[0] = volSet.volumeSize.s[0] - 1;
	volSize[1] = volSet.volumeSize.s[1] - 1;
	volSize[2] = volSet.volumeSize.s[2] - 1;

	maxVolLen = MAX(volSize[0], volSize[1]);
	maxVolLen = MAX(volSize[2], maxVolLen);
	for (maxDepth = 0, maxLenPow2 = 1; maxLenPow2 < maxVolLen;
			maxDepth ++, maxLenPow2 *= 2);
	if (maxDepth >= MAX_DEPTH_COUNT) {
		throw string("volume size too large");
	}

	cubeSize[maxDepth] = 1;

	cubeStart[maxDepth*3]   = (maxLenPow2-volSize[0]) / 2;
	cubeStart[maxDepth*3+1] = (maxLenPow2-volSize[1]) / 2;
	cubeStart[maxDepth*3+2] = (maxLenPow2-volSize[2]) / 2;

	cubeCount[maxDepth*3]   = volSize[0];
	cubeCount[maxDepth*3+1] = volSize[1];
	cubeCount[maxDepth*3+2] = volSize[2];

	for (i = maxDepth - 1; i >= startDepth; i --) {
		cubeSize[i] = cubeSize[i + 1] * 2;

		cubeStart[i*3]   = cubeStart[(i+1)*3] / 2;
		cubeStart[i*3+1] = cubeStart[(i+1)*3+1] / 2;
		cubeStart[i*3+2] = cubeStart[(i+1)*3+2] / 2;

		cubeCount[i*3]   = getCubeCount(cubeStart[(i+1)*3],   cubeCount[(i+1)*3]);
		cubeCount[i*3+1] = getCubeCount(cubeStart[(i+1)*3+1], cubeCount[(i+1)*3+1]);
		cubeCount[i*3+2] = getCubeCount(cubeStart[(i+1)*3+2], cubeCount[(i+1)*3+2]);
	}

#ifdef __VERBOSE
		ostringstream oss;
		oss << endl
			<< "volume file resultion: " << volSet.volumeSize.s[0] << "x"
			<< volSet.volumeSize.s[1] << "x"
			<< volSet.volumeSize.s[2] << endl
			<< "thickness: " << volSet.thickness.s[0] << "x" << volSet.thickness.s[1]
			<< "x" << volSet.thickness.s[2] << endl
			<< "cube coordinate start: " << cubeStart[maxDepth*3] << "x"
			<< cubeStart[maxDepth*3+1] << "x" << cubeStart[maxDepth*3+2] << endl;
#endif

	// set the device constant memory
	checkCudaErrors(
		cudaMemcpyToSymbol(d_cube_count, cubeCount, sizeof(cubeCount))
		, __FILE__, __LINE__);
	checkCudaErrors(
		cudaMemcpyToSymbol(d_cube_start, cubeStart, sizeof(cubeStart))
		, __FILE__, __LINE__);
	checkCudaErrors(
		cudaMemcpyToSymbol(d_cube_size, cubeSize, sizeof(cubeSize))
		, __FILE__, __LINE__);
	checkCudaErrors(
		cudaMemcpyToSymbol(d_slice_thick, volSet.thickness.s, 3 * sizeof(float))
		, __FILE__, __LINE__);
	checkCudaErrors(
		cudaMemcpyToSymbol(d_data_format, &dataFormat, sizeof(dataFormat))
		, __FILE__, __LINE__);
	checkCudaErrors(
		cudaMemcpyToSymbol(d_max_depth, &maxDepth, sizeof(maxDepth))
		, __FILE__, __LINE__);

	/* == traverse the first level of the octree == */

	level_count[startDepth] = cubeCount[startDepth*3]* cubeCount[startDepth*3+1]*
								cubeCount[startDepth*3+2];

    // alloc the node and address array for the first level to evaluate
	checkCudaErrors(cudaMalloc((void**) &d_octLvlPtr[startDepth],
		level_count[startDepth] * sizeof(OctNode)), __FILE__, __LINE__);
	checkCudaErrors(cudaMalloc((void**) &d_childAddrPtr[startDepth],
		level_count[startDepth] * sizeof(unsigned int)), __FILE__, __LINE__);

	dim3 blockDim(1, 1, 1);
	dim3 gridDim(1, 1, 1);
    //getFirstBlockGridDim(cubeCount + startDepth * 3, blockDim, gridDim);
	for (blockDim.x = MAX_BLOCK_DIM; blockDim.x > MIN_BLOCK_DIM; blockDim.x /= 2)
		if (getGridDim(level_count[startDepth], blockDim.x) >= 4)
			break;
	gridDim.x = getGridDim(level_count[startDepth], blockDim.x);

	makeFirstOctLevelKn<<<gridDim, blockDim>>>(
		d_octLvlPtr[startDepth], d_childAddrPtr[startDepth], startDepth,
		d_volData, errorThresh);

	getLastCudaError("Kernel execution failed", __FILE__, __LINE__);

	/* == traverse the left levels of the octree == */

	int depth;
	for (depth = startDepth; depth < maxDepth; depth ++)
	{
#ifdef __GPU_WRITE_BACK
		h_octLvlPtr[depth] = new OctNode[level_count[depth]];
		checkCudaErrors(cudaMemcpy(h_octLvlPtr[depth], d_octLvlPtr[depth],
			sizeof(OctNode)* level_count[depth], cudaMemcpyDeviceToHost), __FILE__, __LINE__);

		h_child_addr[depth] = new unsigned int[level_count[depth]];
		checkCudaErrors(cudaMemcpy(h_child_addr[depth], d_childAddrPtr[depth],
			sizeof(unsigned int)* level_count[depth], cudaMemcpyDeviceToHost), __FILE__, __LINE__);
#endif

		// get the prefix sum of child count to retieve the child address
		thrust::device_ptr<unsigned int> dev_ptr(d_childAddrPtr[depth]);
		thrust::inclusive_scan(dev_ptr, dev_ptr + level_count[depth], dev_ptr);

		getLastCudaError("Kernel execution failed", __FILE__, __LINE__);

#ifdef __GPU_WRITE_BACK
		checkCudaErrors(cudaMemcpy(h_child_addr[depth], d_childAddrPtr[depth],
			sizeof(unsigned int)* level_count[depth], cudaMemcpyDeviceToHost),
			__FILE__, __LINE__);
#endif

		// retrieve the count of nodes in next level
		checkCudaErrors(
			cudaMemcpy(&level_count[depth+1], d_childAddrPtr[depth]+level_count[depth]-1,
			sizeof(unsigned int), cudaMemcpyDeviceToHost), __FILE__, __LINE__);

		if (level_count[depth+1] == 0)
			break;

		// alloc the node and address array for the next level to evaluate
		checkCudaErrors(cudaMalloc((void**)&d_octLvlPtr[depth+1],
			level_count[depth+1]* sizeof(OctNode)), __FILE__, __LINE__);
		if (depth+1 < maxDepth) {
			checkCudaErrors(cudaMalloc((void**)&d_childAddrPtr[depth+1],
				level_count[depth+1]* sizeof(unsigned int)), __FILE__, __LINE__);
		}
		else {
			d_childAddrPtr[depth+1] = NULL;
		}

		blockDim.x = blockDim.y = blockDim.z = 1;
		gridDim.x = gridDim.y = gridDim.z = 1;
		for (blockDim.x = MAX_BLOCK_DIM; blockDim.x > MIN_BLOCK_DIM; blockDim.x /= 2) {
			if (getGridDim(level_count[depth], blockDim.x) >= 4)
				break;
		}
		gridDim.x = getGridDim(level_count[depth], blockDim.x);

		// fill the data into the next level
		makeOctLevelChildKn<<<gridDim, blockDim>>>(
			d_octLvlPtr[depth], d_childAddrPtr[depth], d_octLvlPtr[depth+1],
			d_childAddrPtr[depth+1], level_count[depth], depth, d_volData,
			errorThresh);

#ifdef __GPU_WRITE_BACK
		// wait till the kernel finishes
		checkCudaErrors(cudaDeviceSynchronize(), __FILE__, __LINE__);
#endif

		getLastCudaError("Kernel execution failed", __FILE__, __LINE__);
	}

#ifdef __GPU_WRITE_BACK
	h_octLvlPtr[depth] = new OctNode[level_count[depth]];
	checkCudaErrors(cudaMemcpy(h_octLvlPtr[depth], d_octLvlPtr[depth],
		sizeof(OctNode)* level_count[depth], cudaMemcpyDeviceToHost), __FILE__, __LINE__);
#endif

	// wait till the kernel finishes
	//checkCudaErrors(cudaDeviceSynchronize(), __FILE__, __LINE__);

	// for debug !!
	//checkSolver(h_octLvlPtr, cubeCount[startDepth*3] * cubeCount[startDepth*3+1] * cubeCount[startDepth*3+2]);


#ifdef __VERBOSE
	oss << endl
		<< "octree successfully built" << endl
		<< "node count of each level, " << endl;
	for (depth = startDepth; depth <= maxDepth; depth ++)
		oss << "level " << depth << ": " << level_count[depth] << endl;
	info += oss.str();
#endif
}

void getFirstLevelFaceCount(unsigned int &face_count, const int &start_depth) {
	unsigned short &x_dim = cubeCount[start_depth*3];
	unsigned short &y_dim = cubeCount[start_depth*3+1];
	unsigned short &z_dim = cubeCount[start_depth*3+2];

	face_count = x_dim*y_dim*(z_dim+1) + x_dim*(y_dim+1)*z_dim + (x_dim+1)*y_dim*z_dim;
}

struct splitIsZero {
	__host__ __device__ __forceinline__
	bool operator() (const OctFace &f) {
		return f.split_addr == 0;
	}

	__host__ __device__ __forceinline__
	bool operator() (const thrust::tuple<OctEdge, unsigned int> &t) {
		return thrust::get<1>(t) == 0;
	}
};

struct addFaceSplitAddr {
	__host__ __device__ __forceinline__
	const OctFace& operator() (OctFace &f1, OctFace &f2) {
		f2.split_addr += f1.split_addr;
		return f2;
	}
};

// struct addEdgeSplitAddr {
	// __host__ __device__ __forceinline__
	// const OctEdge& operator() (OctEdge &e1, OctEdge &e2) {
		// e2.split_addr += e1.split_addr;
		// return e2;
	// }
// };

void buildFace(const int startDepth, ___OUT string &info)
{
	ostringstream oss;
	oss << endl << "building face" << endl;

	/* == build the first level face == */

	unsigned int n_splitin_face;
	getFirstLevelFaceCount(n_splitin_face, startDepth);

	dim3 blockDim(1,1,1), gridDim(1,1,1);
	for (blockDim.x = 256; blockDim.x > 32; blockDim.x /= 2)
		if (getGridDim(n_splitin_face, blockDim.x) >= 4)
			break;
	gridDim.x = getGridDim(n_splitin_face, blockDim.x);

	OctFace *d_splitin_face;
	checkCudaErrors(cudaMalloc((void**)&d_splitin_face,
		n_splitin_face* sizeof(OctFace)), __FILE__, __LINE__);

	makeFirstLevelFaceKn<<<gridDim, blockDim>>>(d_splitin_face, startDepth);

#ifdef __GPU_WRITE_BACK
	// wait till the kernel finishes
	checkCudaErrors(cudaDeviceSynchronize(), __FILE__, __LINE__);
#endif

	getLastCudaError("Kernel execution failed", __FILE__, __LINE__);

#ifdef __GPU_WRITE_BACK
	//OctFace *h_face_arr = new OctFace[n_splitin_face];
	//checkCudaErrors(
	//	cudaMemcpy(h_face_arr, d_splitin_face, sizeof(OctFace)* n_splitin_face,
	//	cudaMemcpyDeviceToHost), __FILE__, __LINE__);
	//printArr(h_face_arr, n_splitin_face, "face_arr.txt");
	//printPartialFace(h_face_arr, n_splitin_face, "partial_face.txt");
	//delete[] h_face_arr;
#endif

	int depth;
	for (depth = startDepth; depth < maxDepth; depth ++)
	{
		if (n_splitin_face == 0)
			break;

#ifdef __GPU_WRITE_BACK
		//OctFace *h_face_arr = new OctFace[n_splitin_face];
		//OctFace *h_face_arr2 = new OctFace[n_splitin_face];
#endif

		/* == get the split count of each face == */
		blockDim.x = blockDim.y = blockDim.z = 1;
		gridDim.x = gridDim.y = gridDim.z = 1;
		for (blockDim.x = 256; blockDim.x > 32; blockDim.x /= 2)
			if (getGridDim(n_splitin_face, blockDim.x) >= 4)
				break;
		gridDim.x = getGridDim(n_splitin_face, blockDim.x);

		getFaceSplitCountKn<<<gridDim, blockDim>>>
			(d_splitin_face, d_octLvlPtr[depth], depth, n_splitin_face);

		getLastCudaError("Kernel execution failed", __FILE__, __LINE__);

#ifdef __GPU_WRITE_BACK
		//checkCudaErrors(
		//	cudaMemcpy(h_face_arr, d_splitin_face, sizeof(OctFace)* n_splitin_face,
		//	cudaMemcpyDeviceToHost), __FILE__, __LINE__);
		//printArr(h_face_arr, n_splitin_face, "face_arr.txt");
#endif

		/* == partition the minimal and splittable == */
		thrust::device_ptr<OctFace> thr_face_ptr(d_splitin_face);
		thrust::device_ptr<OctFace> thr_non_zero_ptr =
			thrust::partition(thr_face_ptr, thr_face_ptr + n_splitin_face, splitIsZero());

		unsigned int n_zero = thr_non_zero_ptr - thr_face_ptr;
		unsigned int n_non_zero = n_splitin_face - n_zero;

		/* == alloc minimal face pointer and copy == */
		checkCudaErrors(cudaMalloc((void**)&h_mface_devptr[depth],
			n_zero* sizeof(OctFace)), __FILE__, __LINE__);
		checkCudaErrors(
			cudaMemcpy(h_mface_devptr[depth], d_splitin_face, sizeof(OctFace)* n_zero,
			cudaMemcpyDeviceToDevice), __FILE__, __LINE__);
		n_mface[depth] = n_zero;

#ifdef __GPU_WRITE_BACK
		// usually this line may not be annotated
		h_mface_ptr[depth] = new OctFace[n_zero];
		checkCudaErrors(
			cudaMemcpy(h_mface_ptr[depth], h_mface_devptr[depth], sizeof(OctFace)* n_zero,
			cudaMemcpyDeviceToHost), __FILE__, __LINE__);
		//checkCudaErrors(
		//	cudaMemcpy(h_face_arr, d_splitin_face, sizeof(OctFace)* n_splitin_face,
		//	cudaMemcpyDeviceToHost), __FILE__, __LINE__);
		//partitionCheck(h_face_arr2, n_zero, n_splitin_face);
		//arrEqualCheck(h_face_arr, h_face_arr2, n_splitin_face);

		//checkCudaErrors(
		//	cudaMemcpy(h_face_arr, d_splitin_face, sizeof(OctFace)* n_splitin_face,
		//	cudaMemcpyDeviceToHost), __FILE__, __LINE__);
		//////////////////////////////////////////////////////
		//// for debug
		//int i;
		//for (i = n_zero; i < n_splitin_face; i ++) {
		//	if (h_face_arr[i].level2 == 7 && h_face_arr[i].index2 == 105292)
		//		break;
		//}
#endif

		/* == get the address of splitted face from the count == */

		//OctFace face = *(thr_face_ptr + n_splitin_face - 1);
		//int n_splitted = face.split_addr;

		//face = *thr_non_zero_ptr;
		//face.split_addr = 0;

		//thrust::exclusive_scan(thr_non_zero_ptr, thr_face_ptr + n_splitin_face,
		//	thr_non_zero_ptr, face, addFaceSplitAddr());
		thrust::inclusive_scan(thr_non_zero_ptr, thr_face_ptr + n_splitin_face,
			thr_non_zero_ptr, addFaceSplitAddr());

		OctFace face = *(thr_face_ptr + n_splitin_face - 1);
		// number of splitted
		int n_splitted = face.split_addr;

#ifdef __GPU_WRITE_BACK
		//checkCudaErrors(
		//	cudaMemcpy(h_face_arr2, d_splitin_face, sizeof(OctFace)* n_splitin_face,
		//	cudaMemcpyDeviceToHost), __FILE__, __LINE__);
		//scanCheck(h_face_arr, h_face_arr2, n_splitin_face);
		//arrEqualCheck(h_face_arr, h_face_arr2, n_splitin_face);
		//printArr(h_face_arr, h_face_arr2, n_splitin_face);
		//printArr(h_face_arr2 + n_zero, n_non_zero, "face_arr_non_zero_splitting.txt");
#endif

		/* == get the count of new face incurred by each node in the level == */
		unsigned int *d_newface_addr;
		checkCudaErrors(cudaMalloc((void**)&d_newface_addr,
			level_count[depth]* sizeof(unsigned int)), __FILE__, __LINE__);

		blockDim.x = blockDim.y = blockDim.z = 1;
		gridDim.x = gridDim.y = gridDim.z = 1;
		for (blockDim.x = 256; blockDim.x > 32; blockDim.x /= 2)
			if (getGridDim(level_count[depth], blockDim.x) >= 4)
				break;
		gridDim.x = getGridDim(level_count[depth], blockDim.x);

		getOctLevelNewFaceCountKn<<<gridDim, blockDim>>>
			(d_octLvlPtr[depth], d_newface_addr, level_count[depth]);

		getLastCudaError("Kernel execution failed", __FILE__, __LINE__);

		/* == get the addr of new face iccurred by each node == */
		thrust::device_ptr<unsigned int> thr_newface_addr(d_newface_addr);
		thrust::inclusive_scan
			(thr_newface_addr, thr_newface_addr + level_count[depth], thr_newface_addr);

		unsigned int n_new_face = *(thr_newface_addr + level_count[depth] - 1);

#ifdef __GPU_WRITE_BACK
		//unsigned int *h_newface_addr = new unsigned int[level_count[depth]];
		//checkCudaErrors(
		//	cudaMemcpy(h_newface_addr, d_newface_addr,
		//	sizeof(unsigned int)* level_count[depth], cudaMemcpyDeviceToHost),
		//	__FILE__, __LINE__);
#endif

		/* == alloc splitting in next level and fill with splitted and new == */
		OctFace *d_splitin_nextlvl;
		checkCudaErrors(cudaMalloc((void**)&d_splitin_nextlvl,
			(n_splitted + n_new_face)* sizeof(OctFace)), __FILE__, __LINE__);

		/* == fill the splitted face == */
		if (n_non_zero != 0) {
			blockDim.x = blockDim.y = blockDim.z = 1;
			gridDim.x = gridDim.y = gridDim.z = 1;
			for (blockDim.x = 256; blockDim.x > 64; blockDim.x /= 2)
				if (getGridDim(n_non_zero, blockDim.x) >= 4)
					break;
			gridDim.x = getGridDim(n_non_zero, blockDim.x);

			splitFaceKn<<<gridDim, blockDim>>>
				(d_splitin_face + n_zero, d_splitin_nextlvl, d_octLvlPtr[depth],
				d_childAddrPtr[depth], depth, n_non_zero);

			getLastCudaError("Kernel execution failed", __FILE__, __LINE__);
		}

#ifdef __GPU_WRITE_BACK
		//OctFace *h_splitin_nextlvl = new OctFace[n_splitted + n_new_face];
		//checkCudaErrors(
		//	cudaMemcpy(h_splitin_nextlvl, d_splitin_nextlvl,
		//	sizeof(OctFace)* n_splitted, cudaMemcpyDeviceToHost),
		//	__FILE__, __LINE__);
		//printArr(h_splitin_nextlvl, n_splitted, "face_arr_splitted.txt");
#endif

		/* == fill the new face after the splitted to form the splitting in next level == */
		blockDim.x = blockDim.y = blockDim.z = 1;
		gridDim.x = gridDim.y = gridDim.z = 1;
		for (blockDim.x = 256; blockDim.x > 32; blockDim.x /= 2)
			if (getGridDim(level_count[depth], blockDim.x) >= 4)
				break;
		gridDim.x = getGridDim(level_count[depth], blockDim.x);

		fillNewFaceKn<<<gridDim, blockDim>>>
			(d_octLvlPtr[depth], d_childAddrPtr[depth], d_newface_addr,
			d_splitin_nextlvl + n_splitted, level_count[depth], depth + 1);

		checkCudaErrors(cudaFree(d_splitin_face), __FILE__, __LINE__);
		checkCudaErrors(cudaFree(d_newface_addr), __FILE__, __LINE__);

#ifdef __VERBOSE
		oss << "level " << depth << endl
			<< "\tsplitting face: " << n_splitin_face << endl
			<< "\tminimal face: " << n_zero << endl
			<< "\tsplitted face: " << n_splitted << endl
			<< "\tnew face: " << n_new_face << endl;
#endif

		d_splitin_face = d_splitin_nextlvl;
		n_splitin_face = n_splitted + n_new_face;

#ifdef __GPU_WRITE_BACK
		//checkCudaErrors(
		//	cudaMemcpy(h_splitin_nextlvl, d_splitin_face, sizeof(OctFace)* n_splitin_face,
		//	cudaMemcpyDeviceToHost), __FILE__, __LINE__);
		//printPartialFace(h_splitin_nextlvl, n_splitted, "partial_face_splitted.txt");
		//printPartialFace(h_splitin_nextlvl+n_splitted, n_new_face, "partial_face_new_face.txt");
		//printPartialFace(h_splitin_nextlvl, n_splitin_face, "partial_face.txt");
		//printArr(h_splitin_nextlvl, n_splitin_face, "face_arr.txt");
#endif

#ifdef __GPU_WRITE_BACK
		//delete[] h_face_arr;
		//delete[] h_face_arr2;
		//delete[] h_newface_addr;
		//delete[] h_splitin_nextlvl;
#endif
	}

	if (n_splitin_face != 0) {
#ifdef __VERBOSE
		oss << "level " << depth << ", minimal face: " << n_splitin_face << endl;
#endif
		h_mface_devptr[maxDepth] = d_splitin_face;
		n_mface[maxDepth] = n_splitin_face;

#ifdef __GPU_WRITE_BACK
		h_mface_ptr[maxDepth] = new OctFace[n_mface[maxDepth]];
		checkCudaErrors(
			cudaMemcpy(h_mface_ptr[maxDepth], h_mface_devptr[maxDepth], sizeof(OctFace)* n_mface[maxDepth],
			cudaMemcpyDeviceToHost), __FILE__, __LINE__);
#endif
	}

	info += oss.str();
}

void bldNxtLvlFace(
	const int depth, pOctFace &d_splitin_face, unsigned int &n_splitin_face, ostringstream &oss
){
	// !! I'm wondering if I should free the d_spliliin_face here
	if (n_splitin_face == 0)
		return;


	/* == get the split count of each face == */
	dim3 blockDim, gridDim;
	getLinearBlockGridDim3(n_splitin_face, 256, blockDim, gridDim);
	getFaceSplitCountKn<<<gridDim, blockDim>>>
		(d_splitin_face, d_octLvlPtr[depth], depth, n_splitin_face);

	getLastCudaError("Kernel execution failed", __FILE__, __LINE__);

	/* == get the address of splitted face from the count == */
	thrust::device_ptr<OctFace> thr_face_ptr(d_splitin_face);
	thrust::inclusive_scan(thr_face_ptr, thr_face_ptr + n_splitin_face,
		thr_face_ptr, addFaceSplitAddr());

	OctFace face = *(thr_face_ptr + n_splitin_face - 1);
	// number of splitted
	int n_splitted = face.split_addr;

	/* == get the count of new face incurred by each node in the level == */
	unsigned int *d_newface_addr = NULL;
	checkCudaErrors(cudaMalloc((void**)&d_newface_addr,
		level_count[depth]* sizeof(unsigned int)), __FILE__, __LINE__);

	getLinearBlockGridDim3(level_count[depth], 256, blockDim, gridDim);
	getOctLevelNewFaceCountKn<<<gridDim, blockDim>>>
		(d_octLvlPtr[depth], d_newface_addr, level_count[depth]);

	getLastCudaError("Kernel execution failed", __FILE__, __LINE__);

	/* == get the addr of new face iccurred by each node == */
	thrust::device_ptr<unsigned int> thr_newface_addr(d_newface_addr);
	thrust::inclusive_scan
		(thr_newface_addr, thr_newface_addr + level_count[depth], thr_newface_addr);

	unsigned int n_new_face = *(thr_newface_addr + level_count[depth] - 1);

	/* == alloc splitting in next level == */
	OctFace *d_splitin_nextlvl = NULL;
	checkCudaErrors(cudaMalloc((void**)&d_splitin_nextlvl,
		(n_splitted + n_new_face)* sizeof(OctFace)), __FILE__, __LINE__);

	/* == fill the splitted face == */
	if (n_splitted != 0) {
		getLinearBlockGridDim3(n_splitin_face, 256, blockDim, gridDim);
		splitFaceKn<<<gridDim, blockDim>>>(
			d_splitin_face, d_splitin_nextlvl, d_octLvlPtr[depth], 
			d_childAddrPtr[depth], depth, n_splitin_face);

		getLastCudaError("Kernel execution failed", __FILE__, __LINE__);
	}

	/* == fill the new face after the splitted to form the splitting in next level == */
	getLinearBlockGridDim3(level_count[depth], 256, blockDim, gridDim);
	fillNewFaceKn<<<gridDim, blockDim>>>
		(d_octLvlPtr[depth], d_childAddrPtr[depth], d_newface_addr,
		d_splitin_nextlvl + n_splitted, level_count[depth], depth + 1);

	checkCudaErrors(cudaFree(d_splitin_face), __FILE__, __LINE__);
	checkCudaErrors(cudaFree(d_newface_addr), __FILE__, __LINE__);

#ifdef __VERBOSE
	oss << "\tsplitting face: " << n_splitin_face << endl
		<< "\tsplitted face: " << n_splitted << endl
		<< "\tnew face: " << n_new_face << endl;
#endif

	d_splitin_face = d_splitin_nextlvl;
	n_splitin_face = n_splitted + n_new_face;
}

void bldNxtLvlEdge(
	const int depth, pOctEdge &d_splitin_edge, unsigned int &n_splitin_edge,
	pOctFace &d_splitin_face, unsigned int &n_splitin_face, ostringstream &oss
){
	// !! I'm wondering if I should free the d_spliliin_edge here
	if (n_splitin_edge == 0)
		return;

	/* == alloc the split count/address array == */
	unsigned int *d_edge_split_addr = NULL;
	checkCudaErrors(cudaMalloc((void**)&d_edge_split_addr,
		n_splitin_edge* sizeof(unsigned int)), __FILE__, __LINE__);

	/* == get the split count of each edge == */
	dim3 blockDim, gridDim;
	getLinearBlockGridDim3(n_splitin_edge, 256, blockDim, gridDim);
	getEdgeSplitCountKn<<<gridDim, blockDim>>>(
		d_splitin_edge, d_edge_split_addr, d_octLvlPtr[depth], depth, n_splitin_edge);

	getLastCudaError("Kernel execution failed", __FILE__, __LINE__);

	/* == partition the minimal and splittable edge == */
	thrust::device_ptr<OctEdge> thr_edge_ptr(d_splitin_edge);
	thrust::device_ptr<unsigned int> thr_edge_split_addr(d_edge_split_addr);
	typedef thrust::zip_iterator<
		thrust::tuple< thrust::device_ptr<OctEdge>, thrust::device_ptr<unsigned int> >
		> edge_split_zip_interator;
	edge_split_zip_interator edge_split_iter = thrust::make_zip_iterator(
		thrust::make_tuple(thr_edge_ptr, thr_edge_split_addr));
	edge_split_zip_interator non_zero_iter = thrust::partition(
		edge_split_iter, edge_split_iter + n_splitin_edge, splitIsZero());

	unsigned int n_zero = non_zero_iter - edge_split_iter;
	unsigned int n_non_zero = n_splitin_edge - n_zero;

#ifdef __GPU_WRITE_BACK
	OctEdge *h_splitin_edge = new OctEdge[n_splitin_edge];
	checkCudaErrors(
		cudaMemcpy(h_splitin_edge, d_splitin_edge, n_splitin_edge* sizeof(OctEdge),
		cudaMemcpyDeviceToHost), __FILE__, __LINE__);
#endif

	/* == get the address of splitted edge for splittable edges == */
	thrust::inclusive_scan(
		thr_edge_split_addr + n_zero, thr_edge_split_addr + n_splitin_edge, 
		thr_edge_split_addr + n_zero);
	
	unsigned int n_splitted = *(thr_edge_split_addr + n_splitin_edge - 1);

#ifdef __GPU_WRITE_BACK
	unsigned int *h_edge_split_addr = new unsigned int[n_splitin_edge];
	checkCudaErrors(
		cudaMemcpy(h_edge_split_addr, d_edge_split_addr, n_splitin_edge* sizeof(unsigned int),
		cudaMemcpyDeviceToHost), __FILE__, __LINE__);
#endif

	/* == alloc minimal edge pointer and copy == */
	checkCudaErrors(cudaMalloc((void**)&h_medge_devptr[depth],
		n_zero* sizeof(OctEdge)), __FILE__, __LINE__);
	checkCudaErrors(
		cudaMemcpy(h_medge_devptr[depth], d_splitin_edge, sizeof(OctEdge)* n_zero,
			cudaMemcpyDeviceToDevice), __FILE__, __LINE__);
	n_medge[depth] = n_zero;

#ifdef __GPU_WRITE_BACK
	// copy the minmal edge back to host
	// usually this line may not be annotated
	h_medge_ptr[depth] = new OctEdge[n_zero];
	checkCudaErrors(
		cudaMemcpy(h_medge_ptr[depth], h_medge_devptr[depth], sizeof(OctEdge)* n_zero,
		cudaMemcpyDeviceToHost), __FILE__, __LINE__);
#endif
	
	/* == get the count of new edge incurred by expanding each node in the level == */
	unsigned int *d_node_new_edge_addr = NULL;
	checkCudaErrors(cudaMalloc((void**)&d_node_new_edge_addr,
		level_count[depth]* sizeof(unsigned int)), __FILE__, __LINE__);

	getLinearBlockGridDim3(level_count[depth], 256, blockDim, gridDim);
	getOctLevelNewEdgeCountKn<<<gridDim, blockDim>>>
		(d_octLvlPtr[depth], d_node_new_edge_addr, level_count[depth]);

	getLastCudaError("Kernel execution failed", __FILE__, __LINE__);

	/* == get the addr of new edge iccurred by expanding each node == */
	thrust::device_ptr<unsigned int> thr_node_new_edge_addr(d_node_new_edge_addr);
	thrust::inclusive_scan(
		thr_node_new_edge_addr, thr_node_new_edge_addr + level_count[depth], 
		thr_node_new_edge_addr);

	unsigned int n_node_new_edge = *(thr_node_new_edge_addr + level_count[depth] - 1);
	
#ifdef __GPU_WRITE_BACK
	OctFace *h_splitin_face = new OctFace[n_splitin_face];
	checkCudaErrors(
		cudaMemcpy(h_splitin_face, d_splitin_face, n_splitin_face* sizeof(OctFace),
		cudaMemcpyDeviceToHost), __FILE__, __LINE__);
#endif

	/* == get the count of new edge incurred by splitting each face in the level == */
	unsigned int *d_face_new_edge_addr = NULL;
	checkCudaErrors(cudaMalloc((void**)&d_face_new_edge_addr,
		n_splitin_face* sizeof(unsigned int)), __FILE__, __LINE__);
	
	getLinearBlockGridDim3(n_splitin_face, 256, blockDim, gridDim);
	getFaceNewEdgeCountKn<<<gridDim, blockDim>>>(
		d_splitin_face, d_face_new_edge_addr, depth, d_octLvlPtr[depth], n_splitin_face);
	
	getLastCudaError("Kernel execution failed", __FILE__, __LINE__);
	
	/* == get the addr of new edge iccurred by splitting each face == */
	thrust::device_ptr<unsigned int> thr_face_new_edge_addr(d_face_new_edge_addr);
	thrust::inclusive_scan(
		thr_face_new_edge_addr, thr_face_new_edge_addr + n_splitin_face, 
		thr_face_new_edge_addr);

	unsigned int n_face_new_edge = *(thr_face_new_edge_addr + n_splitin_face - 1);
	
#ifdef __GPU_WRITE_BACK
	unsigned int *h_face_new_addr = new unsigned int[n_splitin_face];
	checkCudaErrors(cudaMemcpy(h_face_new_addr, d_face_new_edge_addr, 
		n_splitin_face *sizeof(unsigned int), cudaMemcpyDeviceToHost), 
		__FILE__, __LINE__);
#endif

	/* == alloc spliting edge array in next level == */
	OctEdge *d_splitin_edge_next_lvl = NULL;
	checkCudaErrors (cudaMalloc((void**)&d_splitin_edge_next_lvl,
		(n_splitted + n_node_new_edge + n_face_new_edge)* sizeof(OctEdge)), 
		__FILE__, __LINE__);
		
	/* == fill the splitted edges == */
	getLinearBlockGridDim3(n_non_zero, 256, blockDim, gridDim);
	splitEdgeKn<<<gridDim, blockDim>>>(
		d_splitin_edge + n_zero, d_splitin_edge_next_lvl, d_edge_split_addr + n_zero, 
		d_octLvlPtr[depth], d_childAddrPtr[depth], depth, n_non_zero);
	
	/* == fill the node new edges == */
	getLinearBlockGridDim3(level_count[depth], 256, blockDim, gridDim);
	fillOctLvlNewEdgeKn<<<gridDim, blockDim>>>(
		d_octLvlPtr[depth], d_childAddrPtr[depth], d_node_new_edge_addr, 
		level_count[depth], depth, d_splitin_edge_next_lvl + n_splitted);
		
	/* == fill the face new edges == */
	////////////////////////
	// for debug !!
	//getLinearBlockGridDim3(200, 256, blockDim, gridDim);
	getLinearBlockGridDim3(n_splitin_face, 256, blockDim, gridDim);
	fillFaceNewEdgeKn<<<gridDim, blockDim>>>(
		d_splitin_face, d_octLvlPtr[depth], d_childAddrPtr[depth], 
		d_face_new_edge_addr, n_splitin_face, depth, 
		d_splitin_edge_next_lvl + n_splitted + n_node_new_edge);
	
	checkCudaErrors(cudaFree(d_splitin_edge), __FILE__, __LINE__);
	checkCudaErrors(cudaFree(d_edge_split_addr), __FILE__, __LINE__);
	checkCudaErrors(cudaFree(d_node_new_edge_addr), __FILE__, __LINE__);
	checkCudaErrors(cudaFree(d_face_new_edge_addr), __FILE__, __LINE__);

#ifdef __CUDA_DBG
	//////////////////////////////////
	// for debug !!
	checkCudaErrors(cudaDeviceSynchronize(), __FILE__, __LINE__);
	writeCuprintf();
#endif

#ifdef __VERBOSE
	oss << "level " << depth << endl
		<< "\tsplitting edge: " << n_splitin_edge << endl
		<< "\tminimal edge: " << n_zero << endl
		<< "\tsplitted edge: " << n_splitted << endl
		<< "\tnode new edge: " << n_node_new_edge << endl
		<< "\tface new edge: " << n_face_new_edge << "," << endl;
#endif
	
	d_splitin_edge = d_splitin_edge_next_lvl;
	n_splitin_edge = n_splitted + n_node_new_edge + n_face_new_edge;

#ifdef __GPU_WRITE_BACK
	OctEdge *h_splitin_edge_nxtl = new OctEdge[n_splitin_edge];
	checkCudaErrors(
		cudaMemcpy(h_splitin_edge_nxtl, d_splitin_edge, n_splitin_edge* sizeof(OctEdge),
		cudaMemcpyDeviceToHost), __FILE__, __LINE__);
#endif

#ifdef __GPU_WRITE_BACK
	delete[] h_splitin_edge;
	delete[] h_edge_split_addr;
	delete[] h_splitin_face;
	delete[] h_face_new_addr;
	delete[] h_splitin_edge_nxtl;
#endif
}

void buildEdge(const int start_depth, ___OUT string &info) 
{
	ostringstream oss;
	oss << endl << "building edge ..." << endl;

	/* == build the first level face == */
	unsigned short &x_dim = cubeCount[start_depth*3];
	unsigned short &y_dim = cubeCount[start_depth*3+1];
	unsigned short &z_dim = cubeCount[start_depth*3+2];

	// first level face count
	unsigned int n_splitin_face = 
		x_dim*y_dim*(z_dim+1) + x_dim*(y_dim+1)*z_dim + (x_dim+1)*y_dim*z_dim;

	// alloc first level splitting face array in GPU
	OctFace *d_splitin_face = NULL;
	checkCudaErrors(cudaMalloc((void**)&d_splitin_face,
		n_splitin_face* sizeof(OctFace)), __FILE__, __LINE__);

	dim3 blockDim, gridDim;
	getLinearBlockGridDim3(n_splitin_face, 256, blockDim, gridDim);
	if (start_depth < maxDepth)
		makeFirstLevelFaceKn<<<gridDim, blockDim>>>(d_splitin_face, start_depth);
	
	/* == build the first level edge == */
	// first level edge count
	unsigned int n_splitin_edge = 
		x_dim*(y_dim+1)*(z_dim+1) + (x_dim+1)*y_dim*(z_dim+1) + (x_dim+1)*(y_dim+1)*z_dim;

	// alloc first level splitting edge array in GPU
	OctEdge *d_splitin_edge = NULL;
	checkCudaErrors(cudaMalloc((void**)&d_splitin_edge,
		n_splitin_edge* sizeof(OctEdge)), __FILE__, __LINE__);

	getLinearBlockGridDim3(n_splitin_edge, 256, blockDim, gridDim);
	makeFirstLevelEdgeKn<<<gridDim, blockDim>>>(d_splitin_edge, start_depth);
	
	int depth;
	for (depth = start_depth; depth < maxDepth; depth ++) {
		if (n_splitin_edge == 0)
			break;
		
		bldNxtLvlEdge(depth, d_splitin_edge, n_splitin_edge, d_splitin_face, 
			n_splitin_face, oss);

		if (depth < maxDepth - 1) {
			bldNxtLvlFace(depth, d_splitin_face, n_splitin_face, oss);
		}
	}
	
	if (n_splitin_edge != 0) {
#ifdef __VERBOSE
		oss << "level " << maxDepth << ", minimal edge: " << n_splitin_edge << endl;
#endif
		h_medge_devptr[maxDepth] = d_splitin_edge;
		n_medge[maxDepth] = n_splitin_edge;

#ifdef __GPU_WRITE_BACK
		// copy the minmal edge back to host
		h_medge_ptr[depth] = new OctEdge[n_medge[depth]];
		checkCudaErrors(
			cudaMemcpy(h_medge_ptr[depth], h_medge_devptr[depth], sizeof(OctEdge)* n_medge[depth],
			cudaMemcpyDeviceToHost), __FILE__, __LINE__);
#endif
	}
	
	checkCudaErrors(cudaFree(d_splitin_face), __FILE__, __LINE__);

	info += oss.str();
}

void genIso(const int startDepth, const float isovalue, ___OUT string &info) {
	unsigned int sum = n_mface[startDepth];
	for (int i = startDepth+1; i <= maxDepth; i ++) {
		sum += n_mface[i];
		n_mface[i] = sum;
	}

	// copy constant to device
	checkCudaErrors(
		cudaMemcpyToSymbol(d_mface_ptr, h_mface_devptr, sizeof(h_mface_devptr))
		, __FILE__, __LINE__);
	checkCudaErrors(
		cudaMemcpyToSymbol(d_n_mface, n_mface, sizeof(n_mface))
		, __FILE__, __LINE__);

	checkCudaErrors(
		cudaMemcpyToSymbol(dev_octlvl_ptr, d_octLvlPtr, sizeof(d_octLvlPtr))
		, __FILE__, __LINE__);
	checkCudaErrors(
		cudaMemcpyToSymbol(d_octlvl_count, level_count, sizeof(level_count))
		, __FILE__, __LINE__);

	unsigned int *d_tri_count;
	checkCudaErrors(cudaMalloc((void**)&d_tri_count,
		sizeof(unsigned int)* n_mface[maxDepth]), __FILE__, __LINE__);

	dim3 gridDim, blockDim;
	blockDim.x = blockDim.y = blockDim.z = 1;
	gridDim.x = gridDim.y = gridDim.z = 1;
	for (blockDim.x = 256; blockDim.x > 32; blockDim.x /= 2)
		if (getGridDim(n_mface[maxDepth], blockDim.x) >= 4)
			break;
	gridDim.x = getGridDim(n_mface[maxDepth], blockDim.x);

	getIsosurfCountKn<<<gridDim, blockDim>>>
		(d_volData, startDepth, isovalue, d_tri_count);

	getLastCudaError("Kernel execution failed", __FILE__, __LINE__);

#ifdef __GPU_WRITE_BACK
	unsigned int *h_tri_count = new unsigned int[n_mface[maxDepth]];
	checkCudaErrors(
		cudaMemcpy(h_tri_count, d_tri_count, sizeof(unsigned int)* n_mface[maxDepth],
		cudaMemcpyDeviceToHost), __FILE__, __LINE__);
#endif

	/* == get the addr of triangles for each minimal face == */
	thrust::device_ptr<unsigned int> thr_tri_addr(d_tri_count);
	thrust::inclusive_scan
		(thr_tri_addr, thr_tri_addr + n_mface[maxDepth], thr_tri_addr);

#ifdef __GPU_WRITE_BACK
	checkCudaErrors(
		cudaMemcpy(h_tri_count, d_tri_count, sizeof(unsigned int)* n_mface[maxDepth],
		cudaMemcpyDeviceToHost), __FILE__, __LINE__);
#endif

	tri_count = *(thr_tri_addr + n_mface[maxDepth] - 1);

	checkCudaErrors(cudaMalloc((void**)&d_tri, sizeof(float)* tri_count* 9),
		__FILE__, __LINE__);

	blockDim.x = blockDim.y = blockDim.z = 1;
	gridDim.x = gridDim.y = gridDim.z = 1;
	for (blockDim.x = 256; blockDim.x > 32; blockDim.x /= 2)
		if (getGridDim(n_mface[maxDepth], blockDim.x) >= 4)
			break;
	gridDim.x = getGridDim(n_mface[maxDepth], blockDim.x);

	genIsosurfKn<<<gridDim, blockDim>>>
		(d_volData, startDepth, isovalue, d_tri_count, d_tri);

	getLastCudaError("Kernel execution failed", __FILE__, __LINE__);

	/////////////////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////
	// for debug
	h_tri = new float[tri_count* sizeof(float)* 9];
	checkCudaErrors(cudaMemcpy(h_tri, d_tri, tri_count* sizeof(float)* 9,
		cudaMemcpyDeviceToHost), __FILE__, __LINE__);

#ifdef __CUDA_DBG
	dbgBufCopyBack();
#endif

	// for debug find the face id of specific triangle
	//unsigned int v;
	//unsigned int* tri_count_ptr = std::upper_bound(h_tri_count, h_tri_count+n_mface[maxDepth], 101769);
	//int fid = tri_count_ptr - h_tri_count;
	//int lfid;
	//int fdepth;
	//for (fdepth = startDepth; fdepth <= maxDepth; fdepth ++) {
	//	if (n_mface[fdepth] > fid) {
	//		lfid = fid - n_mface[fdepth-1];
	//		break;
	//	}
	//}
	//int f_tri_count = *tri_count_ptr - *(tri_count_ptr-1);
	//float* f_tri_ptr = h_tri + (*(tri_count_ptr-1))*9;

	//OctFace f = h_mface_ptr[fdepth][lfid];
	//OctNode node1 = h_octLvlPtr[f.level1][f.index1];
	//OctNode node2 = h_octLvlPtr[f.level2][f.index2];

	//to_Tris(f_tri_ptr, f_tri_count, "_some");
	//to_Tris(h_tri, tri_count);

	//checkZeroTri(h_tri, tri_count);

	//toRawTris(volSet.dataFileName, h_tri, tri_count);
	/////////////////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////

#ifdef __VERBOSE
	ostringstream oss;
	oss << endl << "iso surfaces generated count: " << tri_count << endl;
	info += oss.str();
#endif
}

void genIsoFromEdge(const int startDepth, const float isovalue, ___OUT string &info) 
{
	/* == perform all-prefix-sum on minimal edge count in each level == */
	unsigned int sum = n_medge[startDepth];
	for (int i = startDepth+1; i <= maxDepth; i ++) {
		sum += n_medge[i];
		n_medge[i] = sum;
	}

	/* == copy constant to device == */
	checkCudaErrors(
		cudaMemcpyToSymbol(d_medge_ptr, h_medge_devptr, sizeof(h_medge_devptr))
		, __FILE__, __LINE__);
	checkCudaErrors(
		cudaMemcpyToSymbol(d_n_medge, n_medge, sizeof(n_medge))
		, __FILE__, __LINE__);

	checkCudaErrors(
		cudaMemcpyToSymbol(dev_octlvl_ptr, d_octLvlPtr, sizeof(d_octLvlPtr))
		, __FILE__, __LINE__);
	checkCudaErrors(
		cudaMemcpyToSymbol(d_octlvl_count, level_count, sizeof(level_count))
		, __FILE__, __LINE__);
	
	/* == alloc triangle count/address array for each minimal edge == */
	unsigned int *d_tri_count;
	checkCudaErrors(cudaMalloc((void**)&d_tri_count,
		sizeof(unsigned int)* n_medge[maxDepth]), __FILE__, __LINE__);

	/* == get triangle count for each minimal edge == */
	dim3 blockDim, gridDim;
	//getLinearBlockGridDim3(500, 128, blockDim, gridDim);
	getLinearBlockGridDim3(n_medge[maxDepth], 128, blockDim, gridDim);
	getEdgeIsosurfCountKn<<<gridDim, blockDim>>>
		(d_volData, startDepth, isovalue, d_tri_count);


	////////////////////////////////////////////
	// for debug
	//checkCudaErrors(cudaDeviceSynchronize(), __FILE__, __LINE__);


	getLastCudaError("Kernel execution failed", __FILE__, __LINE__);

#ifdef __GPU_WRITE_BACK
	unsigned int *h_tri_count = new unsigned int[n_medge[maxDepth]];
	checkCudaErrors(
		cudaMemcpy(h_tri_count, d_tri_count, sizeof(unsigned int)* n_medge[maxDepth],
		cudaMemcpyDeviceToHost), __FILE__, __LINE__);
#endif

	/* == get the addr of triangles for each minimal edge == */
	thrust::device_ptr<unsigned int> thr_tri_addr(d_tri_count);
	thrust::inclusive_scan
		(thr_tri_addr, thr_tri_addr + n_medge[maxDepth], thr_tri_addr);

#ifdef __GPU_WRITE_BACK
	checkCudaErrors(
		cudaMemcpy(h_tri_count, d_tri_count, sizeof(unsigned int)* n_medge[maxDepth],
		cudaMemcpyDeviceToHost), __FILE__, __LINE__);
#endif

	tri_count = *(thr_tri_addr + n_medge[maxDepth] - 1);

	checkCudaErrors(cudaMalloc((void**)&d_tri, sizeof(float)* tri_count* 9),
		__FILE__, __LINE__);

	getLinearBlockGridDim3(n_medge[maxDepth], 128, blockDim, gridDim);
	genEdgeIsosurfKn<<<gridDim, blockDim>>>
		(d_volData, startDepth, isovalue, d_tri_count, d_tri);

	getLastCudaError("Kernel execution failed", __FILE__, __LINE__);

#ifdef __VERBOSE
	ostringstream oss;
	oss << endl << "iso surfaces generated count: " << tri_count << endl;
	info += oss.str();
#endif

#ifdef __GPU_WRITE_BACK
	h_tri = new float[tri_count* sizeof(float)* 9];
	checkCudaErrors(cudaMemcpy(h_tri, d_tri, tri_count* sizeof(float)* 9,
		cudaMemcpyDeviceToHost), __FILE__, __LINE__);

	////////////////////////////////////////////////////////
	// for debug find the face id of specific triangle
	//unsigned int v;
	//unsigned int* tri_count_ptr = std::upper_bound(h_tri_count, 
	//	h_tri_count+n_medge[maxDepth], 3480);
	//int eid = tri_count_ptr - h_tri_count;
	//int leid;
	//int edepth;
	//for (edepth = startDepth; edepth <= maxDepth; edepth ++) {
	//	if (n_medge[edepth] > eid) {
	//		leid = eid - n_medge[edepth-1];
	//		break;
	//	}
	//}
	//int e_tri_count = *tri_count_ptr - *(tri_count_ptr-1);
	//float* e_tri_ptr = h_tri + (*(tri_count_ptr-1))*9;

	//OctEdge e = h_medge_ptr[edepth][leid];
	//OctNode node1 = h_octLvlPtr[e.level1][e.index1];
	//OctNode node2 = h_octLvlPtr[e.level2][e.index2];
	//OctNode node3 = h_octLvlPtr[e.level3][e.index3];
	//OctNode node4 = h_octLvlPtr[e.level4][e.index4];

	//to_Tris(e_tri_ptr, e_tri_count, "_some");

	//to_Tris(h_tri, tri_count);

	delete[] h_tri;
	delete[] h_tri_count;
#endif
}

//  Adaptively Generate the Iso-surfaces in Parallel
//  Invoking Cuda Kernel
bool pAdaptiveIso(
    const string& filename, const int startDepth, const float errorThresh,
	const float isovalue, ___OUT string& info
){
	try {
		ostringstream oss;

        /* == set device and timer ==*/
		// use device with highest Gflops/s
		int devID = findCudaDevice();
		std::cout << "Device Id: " << devID << std::endl << std::endl;

		cudaSetDevice(devID);
		cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);

#ifdef __CUDA_DBG
		cudaDeviceProp props;
		checkCudaErrors(cudaGetDeviceProperties(&props, devID), __FILE__, __LINE__);
		//Architectures with compute capability 1.x, function
		//cuPrintf() is used. Otherwise, function printf() is called.
		use_cuPrintf = (props.major < 2);

		if (use_cuPrintf) {
			//Initializaton, allocate buffers on both host
			//and device for data to be printed.
			cudaPrintfInit(100*1024*1024/*20000*256*/);
		}

		checkCudaErrors(cudaMalloc((void**)&h_dbg_buf_devptr, 
			sizeof(hdbg_buf)), __FILE__, __LINE__);
		checkCudaErrors(cudaMemcpyToSymbol(d_dbg_buf, &h_dbg_buf_devptr, 
			sizeof(h_dbg_buf_devptr)), __FILE__, __LINE__);
#endif

        /* == make the volume data ready == */
		StopWatchInterface *total_timer = 0;
		sdkCreateTimer(&total_timer);

#ifdef __VERBOSE
		StopWatchInterface *vol_timer = 0;
		sdkCreateTimer(&vol_timer);

		StopWatchInterface *oct_timer = 0;
		sdkCreateTimer(&oct_timer);

		StopWatchInterface *edge_timer = 0;
		sdkCreateTimer(&edge_timer);

		StopWatchInterface *tri_timer = 0;
		sdkCreateTimer(&tri_timer);

		sdkStartTimer(&vol_timer);
#endif

		sdkStartTimer(&total_timer);

		// read volume file on host
		if (!volSet.parseDataFile(filename, true, false)) {
			info = "cannot open file";
			return false;
		}

		// allocate volume set memory on device
		checkCudaErrors(cudaMalloc((void**) &d_volData, volSet.memSize()), __FILE__, __LINE__);
		// copy host volume data to device
		checkCudaErrors(cudaMemcpy(d_volData, volSet.getData(), volSet.memSize(),
							cudaMemcpyHostToDevice), __FILE__, __LINE__);
		// clear host volume memory
		//volSet.clear();

		if (volSet.format == DATA_CHAR)
			dataFormat = CHAR_TYPE;
		else if (volSet.format == DATA_UCHAR)
			dataFormat = UCHAR_TYPE;
		else if (volSet.format == DATA_SHORT)
			dataFormat = SHORT_TYPE;
		else if (volSet.format == DATA_USHORT)
			dataFormat = USHORT_TYPE;
		else {
            checkCudaErrors(cudaFree(d_volData), __FILE__, __LINE__);
			info = "volume data format unsupported";
			return false;
		}

#ifdef __VERBOSE
		sdkStopTimer(&vol_timer);
#endif

        /* == build octree and edge in parallel == */
#ifdef __VERBOSE
		sdkStartTimer(&oct_timer);
#endif
		buildOctree(startDepth, errorThresh, info);
#ifdef __VERBOSE
		checkCudaErrors(cudaDeviceSynchronize(), __FILE__, __LINE__);
		sdkStopTimer(&oct_timer);
#endif

		//checkOctree(startDepth);

#ifdef __VERBOSE
		sdkStartTimer(&edge_timer);
#endif
		buildEdge(startDepth, info);
#ifdef __VERBOSE
		checkCudaErrors(cudaDeviceSynchronize(), __FILE__, __LINE__);
		sdkStopTimer(&edge_timer);
#endif

#ifdef __CUDA_DBG
		checkEdgeNodeIndexOutBound(startDepth);
#endif

#ifdef __VERBOSE
		sdkStartTimer(&tri_timer);
#endif
		genIsoFromEdge(startDepth, isovalue, info);
#ifdef __VERBOSE
		checkCudaErrors(cudaDeviceSynchronize(), __FILE__, __LINE__);
		sdkStopTimer(&tri_timer);
#endif

		sdkStopTimer(&total_timer);

#ifdef __CUDA_DBG
		if (use_cuPrintf) {
			 writeCuprintf();

			//Free allocated buffers by cudaPrintfInit().
			cudaPrintfEnd();
		}
#endif

		ostringstream oss2;
		oss2 << endl;

#ifdef __VERBOSE
		oss2 << "vol set read copy time: " << sdkGetTimerValue(&vol_timer) / 1000 << "s" << endl
			<< "octree build time: " << sdkGetTimerValue(&oct_timer) / 1000 << "s" << endl
			<< "minimal edge build time: " << sdkGetTimerValue(&edge_timer) / 1000 << "s" << endl
			<< "triangle generate time: " << sdkGetTimerValue(&tri_timer) / 1000 << "s" << endl;

		sdkDeleteTimer(&vol_timer);
		sdkDeleteTimer(&oct_timer);
		sdkDeleteTimer(&edge_timer);
		sdkDeleteTimer(&tri_timer);
#endif
		oss2 << "total time: " << sdkGetTimerValue(&total_timer) / 1000 << "s" << endl;
		sdkDeleteTimer(&total_timer);

		h_tri = new float[tri_count* sizeof(float)* 9];
		checkCudaErrors(cudaMemcpy(h_tri, d_tri, tri_count* sizeof(float)* 9,
			cudaMemcpyDeviceToHost), __FILE__, __LINE__);
		toRawTris(volSet.dataFileName, h_tri, tri_count, oss2);
		
		// for debug 
		//checkZeroTri(h_tri, tri_count);
		//checkTriOutBound(h_tri, 383);

		// free data
        checkCudaErrors(cudaFree(d_volData), __FILE__, __LINE__);

        /* free octree node here */

		cudaDeviceReset();

		info += oss2.str();
	} catch (string& expErrStr) {
		info = expErrStr;
		return false;
	}

	return true;
}
