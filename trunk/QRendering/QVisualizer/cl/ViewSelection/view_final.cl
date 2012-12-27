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

__kernel void viewFinal(
        const uint groupSize, const uint histogramSize,
        __global uint *bufferData,
        __global float *visibilityData,
        __global float *entropyData
         )
{
    uint x = get_global_id(0);
    uint y = get_global_id(1);
    uint size = histogramSize * 2;
    __global uint *data = bufferData + x * 2 + y;
    if (y == 0)
    {
        float visibility = 0.0f;
        for (int i = 0; i < groupSize; i++)
        {
            visibility += convert_float(data[0]) * 0.001f;
            data += size;
        }
        visibilityData[x] = visibility;
    }
    else
    {
        float entropy = 0.0f;
        for (int i = 0; i < groupSize; i++)
        {
            entropy += convert_float(data[0]) * 0.001f;
            data += size;
        }
        entropyData[x] = entropy;
    }
}