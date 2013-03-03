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
#include <sstream>

// includes CUDA
#include <cuda_runtime.h>

// projcet includes
#include "nvtimer.h"
#include "vol_set.h"
#include "pAdapIso.h"

using std::string;

#ifndef MIN
#define MIN(a,b) (a) < (b) ? (a) : (b)
#endif
#ifndef MAX
#define MAX(a,b) (a) > (b) ? (a) : (b)
#endif

__global__ void testKernel( float* g_idata, float* g_odata);

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
void runTest( int argc, char** argv);

////////////////////////////////////////////////////////////////////////////////
// These are CUDA Helper functions

// This will output the proper CUDA error strings in the event that a CUDA host call returns an error
inline void __checkCudaErrors(cudaError err, const char *file, const int line )
{
    if(cudaSuccess != err)
    {
		std::ostringstream stringStream;
		stringStream << file << "(line " << line << ") : CUDA Runtime API error " 
			<< (int)err << ": " << cudaGetErrorString(err) << ".\n";
		std::cerr << stringStream.str();
		throw stringStream.str();
    }

	return;
}

inline void checkCudaErrors(cudaError err)  {
	__checkCudaErrors (err, __FILE__, __LINE__);
}

// This will output the proper error string when calling cudaGetLastError
#define getLastCudaError(msg)      __getLastCudaError (msg, __FILE__, __LINE__)

inline void __getLastCudaError(const char *errorMessage, const char *file, const int line )
{
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err)
    {
        fprintf(stderr, "%s(%i) : getLastCudaError() CUDA error : %s : (%d) %s.\n",
        file, line, errorMessage, (int)err, cudaGetErrorString( err ) );
        exit(-1);
    }
}

inline int _ConvertSMVer2Cores(int major, int minor)
{
    // Defines for GPU Architecture types (using the SM version to determine the # of cores per SM
    typedef struct {
       int SM; // 0xMm (hexidecimal notation), M = SM Major version, and m = SM minor version
       int Cores;
    } sSMtoCores;

    sSMtoCores nGpuArchCoresPerSM[] = 
    { { 0x10,  8 }, // Tesla Generation (SM 1.0) G80 class
      { 0x11,  8 }, // Tesla Generation (SM 1.1) G8x class
      { 0x12,  8 }, // Tesla Generation (SM 1.2) G9x class
      { 0x13,  8 }, // Tesla Generation (SM 1.3) GT200 class
      { 0x20, 32 }, // Fermi Generation (SM 2.0) GF100 class
      { 0x21, 48 }, // Fermi Generation (SM 2.1) GF10x class
      {   -1, -1 }
    };

    int index = 0;
    while (nGpuArchCoresPerSM[index].SM != -1) {
       if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor) ) {
          return nGpuArchCoresPerSM[index].Cores;
       }	
       index++;
    }
    printf("MapSMtoCores undefined SM %d.%d is undefined (please update to the latest SDK)!\n", major, minor);
    return -1;
}

// General GPU Device CUDA Initialization
int gpuDeviceInit(int devID)
{
    int deviceCount;
    checkCudaErrors(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0)
    {
        fprintf(stderr, "gpuDeviceInit() CUDA error: no devices supporting CUDA.\n");
        exit(-1);
    }

    if (devID < 0)
       devID = 0;
        
    if (devID > deviceCount-1)
    {
        fprintf(stderr, "\n");
        fprintf(stderr, ">> %d CUDA capable GPU device(s) detected. <<\n", deviceCount);
        fprintf(stderr, ">> gpuDeviceInit (-device=%d) is not a valid GPU device. <<\n", devID);
        fprintf(stderr, "\n");
        return -devID;
    }

    cudaDeviceProp deviceProp;
    checkCudaErrors( cudaGetDeviceProperties(&deviceProp, devID) );

    if (deviceProp.major < 1)
    {
        fprintf(stderr, "gpuDeviceInit(): GPU device does not support CUDA.\n");
        exit(-1);                                                  
    }
    
    checkCudaErrors( cudaSetDevice(devID) );
    printf("gpuDeviceInit() CUDA Device [%d]: \"%s\n", devID, deviceProp.name);

    return devID;
}

// This function returns the best GPU (with maximum GFLOPS)
int gpuGetMaxGflopsDeviceId()
{
    int current_device     = 0, sm_per_multiproc  = 0;
    int max_compute_perf   = 0, max_perf_device   = 0;
    int device_count       = 0, best_SM_arch      = 0;
    cudaDeviceProp deviceProp;
    cudaGetDeviceCount( &device_count );
    
    // Find the best major SM Architecture GPU device
    while (current_device < device_count)
    {
        cudaGetDeviceProperties( &deviceProp, current_device );
        if (deviceProp.major > 0 && deviceProp.major < 9999)
        {
            best_SM_arch = MAX(best_SM_arch, deviceProp.major);
        }
        current_device++;
    }

    // Find the best CUDA capable GPU device
    current_device = 0;
    while( current_device < device_count )
    {
        cudaGetDeviceProperties( &deviceProp, current_device );
        if (deviceProp.major == 9999 && deviceProp.minor == 9999)
        {
            sm_per_multiproc = 1;
        }
        else
        {
            sm_per_multiproc = _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor);
        }
        
        int compute_perf  = deviceProp.multiProcessorCount * sm_per_multiproc * deviceProp.clockRate;
        
    if( compute_perf  > max_compute_perf )
    {
            // If we find GPU with SM major > 2, search only these
            if ( best_SM_arch > 2 )
            {
                // If our device==dest_SM_arch, choose this, or else pass
                if (deviceProp.major == best_SM_arch)
                {
                    max_compute_perf  = compute_perf;
                    max_perf_device   = current_device;
                 }
            }
            else
            {
                max_compute_perf  = compute_perf;
                max_perf_device   = current_device;
             }
        }
        ++current_device;
    }
    return max_perf_device;
}


// Initialization code to find the best CUDA Device
int findCudaDevice(/*int argc, const char **argv*/)
{
    cudaDeviceProp deviceProp;
    int devID = 0;
    // If the command-line has a device number specified, use it
    //if (checkCmdLineFlag(argc, argv, "device"))
    //{
    //    devID = getCmdLineArgumentInt(argc, argv, "device=");
    //    if (devID < 0)
    //    {
    //        printf("Invalid command line parameter\n ");
    //        exit(-1);
    //    }
    //    else
    //    {
    //        devID = gpuDeviceInit(devID);
    //        if (devID < 0)
    //        {
    //            printf("exiting...\n");
    //            shrQAFinishExit(argc, (const char **)argv, QA_FAILED);
    //            exit(-1);
    //        }
    //    }
    //}
    //else
    //{
        // Otherwise pick the device with highest Gflops/s
        devID = gpuGetMaxGflopsDeviceId();
        checkCudaErrors( cudaSetDevice( devID ) );
        checkCudaErrors( cudaGetDeviceProperties(&deviceProp, devID) );
        printf("GPU Device %d: \"%s\" with compute capability %d.%d\n\n", devID, deviceProp.name, deviceProp.major, deviceProp.minor);
    //}
    return devID;
}
// end of CUDA Helper Functions


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
		typedef struct {
			unsigned int x, y, z;
		} UINT3;
		const unsigned int MAX_DEPTH_COUNT = 11;
		unsigned int maxVolLen, maxLenPow2, maxDepth, cubeSize[MAX_DEPTH_COUNT];
		UINT3 cubeStart[MAX_DEPTH_COUNT], cubeCount[MAX_DEPTH_COUNT], volSize;
		int i;
		
		volSize.x = volSet.volumeSize.[0] - 1;
		volSize.y = volSet.volumeSize.[1] - 1;
		volSize.z = volSet.volumeSize.[2] - 1;

		maxVolLen = MAX(volSize.x, volSize.y);
		maxVolLen = MAX(volSize.z, maxVolLen);
		for (maxDepth = 0, maxLenPow2 = 1; maxLenPow2 < maxVolLen; maxDepth ++, maxLenPow2 *= 2);
		if (maxDepth >= MAX_DEPTH_COUNT) {
			errorStr = "volume size too large";
			return false;
		}

		cubeSize[maxDepth] = 1;
		for (i = maxDepth - 1; i >= 0; i --) 
			cubeSize[i] = cubeSize[i + 1] * 2;

		cubeStart[maxDepth].x = (maxLenPow2 - volSize.x) / 2;
		cubeStart[maxDepth].y = (maxLenPow2 - volSize.y) / 2;
		cubeStart[maxDepth].z = (maxLenPow2 - volSize.z) / 2;

		cubeCount[maxDepth].x = volSize.x;
		cubeCount[maxDepth].y = volSize.y;
		cubeCount[maxDepth].z = volSize.z;

		for (i = maxDepth - 1; i >= 0; i --) {
			cubeStart[i].x = cubeStart[i + 1].x / 2;
			cubeStart[i].y = cubeStart[i + 1].y / 2;
			cubeStart[i].z = cubeStart[i + 1].z / 2;

			if (cubeStart[i + 1].x % 2 == 0) 
				cubeCount[i].x = cubeCount[i + 1].x / 2;
			else
				cubeCount[i].x = (cubeCount[i + 1].x - 1) / 2 + 1;
			
			if (cubeStart[i + 1].y % 2 == 0) 
				cubeCount[i].y = cubeCount[i + 1].y / 2;
			else
				cubeCount[i].y = (cubeCount[i + 1].y - 1) / 2 + 1;
			
			if (cubeStart[i + 1].z % 2 == 0) 
				cubeCount[i].z = cubeCount[i + 1].z / 2;
			else
				cubeCount[i].z = (cubeCount[i + 1].z - 1) / 2 + 1;
		}


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
