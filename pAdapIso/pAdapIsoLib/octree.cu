#ifndef _OCTREE_KERNEL_H_
#define _OCTREE_KERNEL_H_

#include <stdio.h>

////////////////////////////////////////////////////////////////////////////////
//! Simple test kernel for device functionality
//! @param g_idata  input data in global memory
//! @param g_odata  output data in global memory
////////////////////////////////////////////////////////////////////////////////
__global__ void
testKernel( float* g_idata, float* g_odata) 
{
  // shared memory
  // the size is determined by the host application
  extern  __shared__  float sdata[];

  // access thread id
  const unsigned int tid = threadIdx.x;
  // access number of threads in this block
  const unsigned int num_threads = blockDim.x;

  // read in input data from global memory
  sdata[tid] = g_idata[tid];
  __syncthreads();

  // perform some computations
  sdata[tid] = (float) num_threads * sdata[tid];
  __syncthreads();

  // write data to global memory
  g_odata[tid] = sdata[tid];
}

// kernel for traversing first octree level
__global__ void
travFirstOctLvlKn( OctNode* level_ptr, int depth, unsigned int* child_count, 
				   char* vol_data, float iso_value, float error_thresh ) {
	const unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid < d_cube_count[depth * 3] * d_cube_count[depth * 3 + 1] * 
		d_cube_count[depth * 3 + 2]) {
		OctNode node = level_ptr[tid];
	}
}

#endif // #ifndef _OCTREE_KERNEL_H_
