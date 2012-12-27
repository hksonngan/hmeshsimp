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
 * @date    2012/04/08
 */

__kernel void volumeGradient(__global float4 *volumeData, const uint4 volumeSize, const uint volumePass)
{
    uint x = get_global_id(0);
    uint y = get_global_id(1);
    uint z = volumePass;
    if (x >= volumeSize.x || y >= volumeSize.y || z >= volumeSize.z) return;
    
    const uint4 offset = (uint4)(1, volumeSize.x, volumeSize.x * volumeSize.y, 0);
    __global float4 *data = volumeData + x + y * offset.y + z * offset.z;

    const uint4 index = (uint4)(x, y, z, 0);
    const float4 indexf = convert_float4(index);
    const int4 previous = convert_int4(max(indexf - 1.0f, (float4)(0.0f)));
    const int4 next = convert_int4(min(indexf + 1.0f, convert_float4(volumeSize - 1)));

    float3 gradient = (float3)(
        volumeData[ next.x + index.y * offset.y + index.z * offset.z].w - volumeData[previous.x +    index.y * offset.y +    index.z * offset.z].w,
        volumeData[index.x +  next.y * offset.y + index.z * offset.z].w - volumeData[   index.x + previous.y * offset.y +    index.z * offset.z].w,
        volumeData[index.x + index.y * offset.y +  next.z * offset.z].w - volumeData[   index.x +    index.y * offset.y + previous.z * offset.z].w
        );

    data[0].xyz = normalize(gradient);
}