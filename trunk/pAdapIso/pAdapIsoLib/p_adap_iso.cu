/*
 *  Adaptively Generate the Iso-surfaces in Parallel
 *  Run On the Host Invoking the Cuda Kernel
 *
 *  Author: Ht
 *  Email : waytofall916 at gmail dot com
 *
 *  Copyright (C) Ht. All rights reserved.
 */

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <string>
#include <iostream>

// includes CUDA
#include <cuda_runtime.h>

// projcet includes
#include "nvtimer.h"
#include "vol_set.h"
#include "p_adap_iso.h"

const unsigned int MAX_DEPTH_COUNT = 11;

// global cuda variables decalaration
__constant__ short d_cube_count[MAX_DEPTH_COUNT * 3], d_cube_start[MAX_DEPTH_COUNT * 3];
__constant__ int d_data_format, d_max_depth;

// cuda type decalaration
typedef struct _OctNode {
	short cube_index[3];
	(struct _OctNode*) children[8];
} OctNode;

// other cuda files
#include "cuda_helper.cu"
#include "octree.cu"

using std::string;

#ifndef MIN
#define MIN(a,b) (a) < (b) ? (a) : (b)
#endif
#ifndef MAX
#define MAX(a,b) (a) > (b) ? (a) : (b)
#endif

unsigned int getCubeCount(unsigned int lowerLayerStart, unsigned int lowerLayerCount) {
	if (lowerLayerStart % 2 == 0) 
		return lowerLayerCount / 2;
	else
		return (lowerLayerCount - 1) / 2 + 1;
}

unsigned int getGridDim(unsigned int validThreadCount, unsigned int blockDim) {
	if (validthreadCount % blockDim == 0)
		return validThreadCount / blockDim;
	else
		return validThreadCount / blockDim + 1;
}

////////////////////////////////////////////////////////////////////////////////
// Adaptively Generate the Iso-surfaces in Parallel
// Invoking Cuda Kernel
////////////////////////////////////////////////////////////////////////////////
bool pAdaptiveIso(const string& filename, int startDepth, float errorThresh, ___OUT string& errorStr)
{
	try {
		// use device with highest Gflops/s
		int devID = findCudaDevice();

		StopWatchInterface *timer = 0;
		sdkCreateTimer( &timer );
		sdkStartTimer( &timer );

		// read volume file on host
		VolumeSet volSet;
		if (!volSet.parseDataFile(filename, true, false)) {
			errorStr = "cannot open file";
			return false;
		}
		// allocate volume set memory on device
		char* d_volData;
		checkCudaErrors( cudaMalloc( (void**) &d_volData, volSet.memSize() ) );
		// copy host volume data to device
		checkCudaErrors( cudaMemcpy( d_volData, volSet.getData(), volSet.memSize(),
							cudaMemcpyHostToDevice) );
		// clear host volume memory
		volSet.clear();

		// compute the cube size, depth, start cordinate for the octree in each depth
		unsigned int maxVolLen, maxLenPow2, maxDepth, cubeSize[MAX_DEPTH_COUNT];
		short cubeStart[MAX_DEPTH_COUNT * 3], cubeCount[MAX_DEPTH_COUNT * 3], volSize[3];
		(OctNode*) d_octLvlPtr[MAX_DEPTH_COUNT];
		unsigned int *d_childAddr;
		int i;
		
		volSize[0] = volSet.volumeSize.s[0] - 1;
		volSize[1] = volSet.volumeSize.s[1] - 1;
		volSize[2] = volSet.volumeSize.s[2] - 1;

		maxVolLen = MAX(volSize.x, volSize.y);
		maxVolLen = MAX(volSize.z, maxVolLen);
		for (maxDepth = 0, maxLenPow2 = 1; maxLenPow2 < maxVolLen; maxDepth ++, maxLenPow2 *= 2);
		if (maxDepth >= MAX_DEPTH_COUNT) {
			errorStr = "volume size too large";
			return false;
		}

		cubeSize[maxDepth] = 1;

		cubeStart[maxDepth * 3]     = (maxLenPow2 - volSize[0]) / 2;
		cubeStart[maxDepth * 3 + 1] = (maxLenPow2 - volSize[1]) / 2;
		cubeStart[maxDepth * 3 + 2] = (maxLenPow2 - volSize[2]) / 2;

		cubeCount[maxDepth * 3]     = volSize[0];
		cubeCount[maxDepth * 3 + 1] = volSize[1];
		cubeCount[maxDepth * 3 + 2] = volSize[2];

		for (i = maxDepth - 1; i >= startDepth; i --) {
			cubeSize[i] = cubeSize[i + 1] * 2;

			cubeStart[i * 3]     = cubeStart[(i + 1) * 3] / 2;
			cubeStart[i * 3 + 1] = cubeStart[(i + 1) * 3 + 1] / 2;
			cubeStart[i * 3 + 2] = cubeStart[(i + 1) * 3 + 2] / 2;

			cubeCount[i * 3]     = 
				getCubeCount(cubeStart[(i + 1) * 3],     cubeCount[(i + 1) * 3]);
			cubeCount[i * 3 + 1] = 
				getCubeCount(cubeStart[(i + 1) * 3 + 1], cubeCount[(i + 1) * 3 + 1]);
			cubeCount[i * 3 + 2] = 
				getCubeCount(cubeStart[(i + 1) * 3 + 2], cubeCount[(i + 1) * 3 + 2]);
		}

		// traverse the first level of the octree
		checkCudaErrors( cudaMalloc( (void**) &d_octLvlPtr[startDepth], 
			cubeCount[startDepth * 3] * cubeCount[startDepth * 3 + 1] * 
			cubeCount[startDepth * 3 + 2] ) );
		checkCudaErrors( cudaMalloc( (void**) &d_childAddr, 
			cubeCount[startDepth * 3] * cubeCount[startDepth * 3 + 1] * 
			cubeCount[startDepth * 3 + 2] ) );
		dim3  blockDim( 32, 8, 1);
		dim3  gridDim( getGridDim(cubeCount[startDepth * 3], blockDim.x), 
				getGridDim(cubeCount[startDepth * 3 + 1], blockDim.y), 
				getGridDim(cubeCount[startDepth * 3 + 2], blockDim.z) );
		travFirstOctLvlKn<<< gridDim, blockDim >>>( d_octLvlPtr[startDepth], startDepth, d_childAddr, errorThresh );
		


		////////////////////////////////////////////////////////
		unsigned int num_threads = 32;
		unsigned int mem_size = sizeof( float) * num_threads;

		// allocate host memory
		float* h_idata = (float*) malloc( mem_size);
		// initalize the memory
		for( unsigned int i = 0; i < num_threads; ++i) 
		{
			h_idata[i] = (float) i;
		}

		// allocate device memory
		float* d_idata;
		checkCudaErrors( cudaMalloc( (void**) &d_idata, mem_size) );
		// copy host memory to device
		checkCudaErrors( cudaMemcpy( d_idata, h_idata, mem_size,
									cudaMemcpyHostToDevice) );

		// allocate device memory for result
		float* d_odata;
		checkCudaErrors( cudaMalloc( (void**) &d_odata, mem_size) );

		// setup execution parameters
		dim3  grid( 1, 1, 1);
		dim3  threads( num_threads, 1, 1);

		// execute the kernel
		testKernel<<< grid, threads, mem_size >>>( d_idata, d_odata);

		// check if kernel execution generated and error
		getLastCudaError("Kernel execution failed");

		// allocate mem for the result on host side
		float* h_odata = (float*) malloc( mem_size);
		// copy result from device to host
		checkCudaErrors( cudaMemcpy( h_odata, d_odata, sizeof( float) * num_threads,
									cudaMemcpyDeviceToHost) );

		sdkStopTimer( &timer );
		printf( "Processing time: %f (ms)\n", sdkGetTimerValue( &timer ) );
		sdkDeleteTimer( &timer );

		// cleanup memory
		free( h_idata );
		free( h_odata );
		checkCudaErrors(cudaFree(d_idata));
		checkCudaErrors(cudaFree(d_odata));

		/// annote it off for v4.0 above!!!
		//cudaDeviceReset();
	} catch (string& expErrStr) {
		errorStr = expErrStr;
		return false;
	}

	return true;
}
