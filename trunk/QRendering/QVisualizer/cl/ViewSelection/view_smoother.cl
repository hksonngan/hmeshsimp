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
 * @date    2012/04/10
 */

#define EPSILON         1e-5

#if GAUSSIAN_2D == 7
    __constant float gaussian2[49] =
    {
        0.0030f, 0.0068f, 0.0111f, 0.0130f, 0.0111f, 0.0068f, 0.0030f,
        0.0068f, 0.0154f, 0.0251f, 0.0295f, 0.0251f, 0.0154f, 0.0068f,
        0.0111f, 0.0251f, 0.0409f, 0.0482f, 0.0409f, 0.0251f, 0.0111f,
        0.0130f, 0.0295f, 0.0482f, 0.0567f, 0.0482f, 0.0295f, 0.0130f,
        0.0111f, 0.0251f, 0.0409f, 0.0482f, 0.0409f, 0.0251f, 0.0111f,
        0.0068f, 0.0154f, 0.0251f, 0.0295f, 0.0251f, 0.0154f, 0.0068f,
        0.0030f, 0.0068f, 0.0111f, 0.0130f, 0.0111f, 0.0068f, 0.0030f
    };
#elif GAUSSIAN_2D == 5
    __constant float gaussian2[25] =
    {
        0.0085f, 0.0223f, 0.0307f, 0.0223f, 0.0085f,
        0.0223f, 0.0583f, 0.0802f, 0.0583f, 0.0223f,
        0.0307f, 0.0802f, 0.1105f, 0.0802f, 0.0307f,
        0.0223f, 0.0583f, 0.0802f, 0.0583f, 0.0223f,
        0.0085f, 0.0223f, 0.0307f, 0.0223f, 0.0085f
    };
#elif GAUSSIAN_2D == 3
    __constant float gaussian2[9] =
    {
        0.0509f, 0.1238f, 0.0509f,
        0.1238f, 0.3012f, 0.1238f,
        0.0509f, 0.1238f, 0.0509f
    };
#else
    __constant float gaussian2[1] =
    {
        1.0000f
    };
#endif

__constant float4 cmin = (float4)(0.0f);
__constant float4 cmax = (float4)(1.0f);

uint float4ToInt(float4 c)
{
    const uint4 color = convert_uint4_rtn(clamp(c, cmin, cmax) * 255);
    return (color.w << 24) | (color.z << 16) | (color.y << 8) | (color.x);
}

__kernel void volumeSmoothing(
    __global float4 *colorbuffer,
    __global uint *pixelBuffer
    )
{
    const uint2 index = (uint2)(get_global_id(0), get_global_id(1));
    const uint2 size = (uint2)(get_global_size(0), get_global_size(1));
    const float2 indexf = convert_float2(index);
    const float2 indexfStart = indexf - GAUSSIAN_2D / 2;
    const float2 indexfMin = (float2)(0.0f);
    const float2 indexfMax = convert_float2(size - 1);

    uint2 indices[GAUSSIAN_2D];
    for (int i = 0; i < GAUSSIAN_2D; i++)
        indices[i] = convert_uint2(clamp(indexfStart + i, indexfMin, indexfMax));

    float4 color = (float4)(0.0f);
    for (int i = 0; i < GAUSSIAN_2D; i++)
        for (int j = 0; j < GAUSSIAN_2D; j++)
            color += colorbuffer[indices[i].x + indices[j].y * size.x] * gaussian2[i + j * GAUSSIAN_2D];

    pixelBuffer[index.x + index.y * size.x] = float4ToInt(color);
}