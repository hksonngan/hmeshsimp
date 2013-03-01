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

// includes CUDA
#include <cuda_runtime.h>

// projcet includes
#include "nvtimer.h"
#include "vol_set.h"

#define ___OUT

using std::string;

#ifndef MIN
#define MIN(a,b) ((a < b) ? a : b)
#endif
#ifndef MAX
#define MAX(a,b) ((a > b) ? a : b)
#endif

__global__ void testKernel( float* g_idata, float* g_odata);

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
void runTest( int argc, char** argv);

////////////////////////////////////////////////////////////////////////////////
// These are CUDA Helper functions

// This will output the proper CUDA error strings in the event that a CUDA host call returns an error
#define checkCudaErrors(err, errstr)  __checkCudaErrors (err, errstr, __FILE__, __LINE__)

inline bool __checkCudaErrors(cudaError err, string &errStr const char *file, const int line )
{
    if(cudaSuccess != err)
    {
        fprintf(stderr, "%s(%i) : CUDA Runtime API error %d: %s.\n", file, line, (int)err, cudaGetErrorString( err ) );
		errStr = cudaGetErrorString(err);
        //exit(-1);
		return false;
    }

	return true;
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
	// use device with highest Gflops/s
	int devID = findCudaDevice();

    StopWatchInterface *timer = 0;
    sdkCreateTimer( &timer );
    sdkStartTimer( &timer );

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
    checkCudaErrors( cudaMalloc( (void**) &d_idata, mem_size), errorStr );
    // copy host memory to device
    checkCudaErrors( cudaMemcpy( d_idata, h_idata, mem_size,
                                cudaMemcpyHostToDevice) , errorStr );

    // allocate device memory for result
    float* d_odata;
    checkCudaErrors( cudaMalloc( (void**) &d_odata, mem_size), errorStr );

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
                                cudaMemcpyDeviceToHost), errorStr );

    sdkStopTimer( &timer );
    printf( "Processing time: %f (ms)\n", sdkGetTimerValue( &timer ) );
    sdkDeleteTimer( &timer );

    // cleanup memory
    free( h_idata );
    free( h_odata );
    checkCudaErrors(cudaFree(d_idata), errorStr);
    checkCudaErrors(cudaFree(d_odata), errorStr);

    cudaDeviceReset();
}
