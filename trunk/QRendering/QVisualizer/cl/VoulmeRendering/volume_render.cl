/**
 * COPYRIGHT NOTICE
 * Copyright (c) 2012, Institute of CG & CAD, Tsinghua University.
 * All Rights Reserved.
 * 
 * @file    *.cl
 * @brief   * functions definition.
 * 
 * This file defines *.
 * 
 * @version 1.0
 * @author  Jackie Pang
 * @e-mail: 15pengyi@gmail.com
 * @date    2012/02/07
 */

#define EPSILON 1e-5

// intersect ray with a box
// http://www.siggraph.org/education/materials/HyperGraph/raytrace/rtinter3.htm

int rayBoxIntersection(float4 box, float4 r_o, float4 r_d, float *tnear, float *tfar)
{
    // compute intersection of ray with all six bbox planes
    const float4 invR = (float4)(1.0f, 1.0f, 1.0f, 1.0f) / r_d;
    const float4 tbot = invR * (-box - r_o);
    const float4 ttop = invR * (box - r_o);

    // re-order intersections to find smallest and largest on each axis
    const float4 tmin = min(ttop, tbot);
    const float4 tmax = max(ttop, tbot);

    // find the largest tmin and the smallest tmax
    const float largest_tmin = max(max(tmin.x, tmin.y), tmin.z);
    const float smallest_tmax = min(min(tmax.x, tmax.y), tmax.z);

	*tnear = largest_tmin;
	*tfar = smallest_tmax;

	return smallest_tmax > largest_tmin;
}

uint float4ToInt(float4 c)
{
    const float4 cmin = (float4)(0.0f);
    const float4 cmax = (float4)(1.0f);
    const uint4 color = convert_uint4(clamp(c, cmin, cmax) * 255.0f);

    return (color.w << 24) | (color.z << 16) | (color.y << 8) | (color.x);
}

float4 backProjection(__global float* m, float4 v)
{
    return (float4)(
        dot(v, (float4)(m[0], m[4], m[ 8], m[12])),
        dot(v, (float4)(m[1], m[5], m[ 9], m[13])),
        dot(v, (float4)(m[2], m[6], m[10], m[14])),
        1.0f
        );
}

__kernel void volumeRendering(
        __global float *inverseViewMatrix,
        __global uint *pixelBuffer,
        __global float *volume, const uint volumeOrigin, const uint4 volumeSize,
        //__read_only image3d_t volume, __global float *volume,
        __read_only image2d_t transferFunction, sampler_t transferFunctionSampler,
        const uint imageWidth, const uint imageHeight,
        const float volumeOffset, const float volumeScale, const float volumeStepSize, const float4 boxSize
#ifdef __CL_ENABLE_DEBUG
        ,__global float *debugOutput
#endif
         )
{
    uint x = get_global_id(0);
    uint y = get_global_id(1);
    if (x >= imageWidth || y >= imageHeight) return;

    uint i = y * imageWidth + x;
    uint imageSize = min(imageWidth, imageHeight);
    float u = imageSize == 1 ? 0.0f : ((float)x * 2.0f - imageWidth + 1) / (imageSize - 1);
    float v = imageSize == 1 ? 0.0f : ((float)y * 2.0f - imageHeight + 1) / (imageSize - 1);
    
    // calculate eye ray in world space
    float4 rayOrigin  = backProjection(inverseViewMatrix, (float4)(0.0f, 0.0f, 0.0f, 1.0f));
    float4 rayDirection = backProjection(inverseViewMatrix, normalize((float4)(u, v, -4.0f, 0.0f)));
    
    // find intersection with box
	float tnear, tfar;
    if (!rayBoxIntersection(boxSize, rayOrigin , rayDirection, &tnear, &tfar))
    {
        // write output color
        pixelBuffer[i] = 0;
        return;
    }
	if (tnear < 0.0f) tnear = 0.0f;     // clamp to near plane

    // march along ray from back to front, accumulating color
    float4 sampleOffset = convert_float4(volumeSize - 1) * 0.5;
    float4 sampleScale = sampleOffset / boxSize;
    float4 sampleOrigin = rayOrigin * sampleScale + sampleOffset;
    float4 sampleStepSize = rayDirection * sampleScale * volumeStepSize;
    float4 samplePosition = rayDirection * sampleScale * tnear + sampleOrigin;
    float4 pixelColor = (float4)(0.0f);
    float2 scalar = (float2)(0.5f);
    uint4 offset = (uint4)(1, volumeSize.x, volumeSize.x * volumeSize.y, volumeSize.x * volumeSize.y * volumeSize.z);
    for(float t = tnear; t < tfar; t += volumeStepSize)
    {
        // read from 3D texture
        uint4 sampleFloor = convert_uint4(round(samplePosition));
        scalar.x = (volume[volumeOrigin + sampleFloor.x * offset.x + sampleFloor.y * offset.y + sampleFloor.z * offset.z] + volumeOffset) * volumeScale;
        
#ifdef __CL_ENABLE_DEBUG
        if (x == imageWidth / 2 && y == imageHeight / 2)
        {
            debugOutput[0] = 0.5f;
        }
#endif

        // lookup in transfer function texture
        float4 color = read_imagef(transferFunction, transferFunctionSampler, scalar);

        // accumulate result
        pixelColor += (float4)(color.xyz, 1.0) * color.w * (1.0 - pixelColor.w);

        // early ray termination
        if (pixelColor.w >= 1.0) break;

        // move to the next one
        samplePosition += sampleStepSize;
    }
    
    // march along ray from back to front, accumulating color
    pixelBuffer[i] = float4ToInt(pixelColor);
}