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
 * @date    2012/04/14
 */

__constant uchar numberOfTriangles[256] =
{
    0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 2, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 3, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 3, 2, 3, 3, 2, 3, 4, 4, 3, 3, 4, 4, 3, 4, 5, 5, 2, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 3, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 4, 2, 3, 3, 4, 3, 4, 2, 3, 3, 4, 4, 5, 4, 5, 3, 2, 3, 4, 4, 3, 4, 5, 3, 2, 4, 5, 5, 4, 5, 2, 4, 1, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 3, 2, 3, 3, 4, 3, 4, 4, 5, 3, 2, 4, 3, 4, 3, 5, 2, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 4, 3, 4, 4, 3, 4, 5, 5, 4, 4, 3, 5, 2, 5, 4, 2, 1, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 2, 3, 3, 2, 3, 4, 4, 5, 4, 5, 5, 2, 4, 3, 5, 4, 3, 2, 4, 1, 3, 4, 4, 5, 4, 5, 3, 4, 4, 5, 5, 2, 3, 4, 2, 1, 2, 3, 3, 2, 3, 4, 2, 1, 3, 2, 4, 1, 2, 1, 1, 0
};

float getValue(
    __global float* volumeData,
    uint4 size,
    uint x,
    uint y,
    uint z
    )
{
    if (x >= size.x) x = size.x - 1;
    if (y >= size.y) y = size.y - 1;
    if (z >= size.z) z = size.z - 1;
    return volumeData[x + y * size.x + z * size.x * size.y];
}

__kernel void classifyCubes(
    __global uchar* histogram,
    __global float* data,
    __private float isoValue
    )
{
    uint4 index  = { get_global_id(0), get_global_id(1), get_global_id(2), 0 };
    uint4 size   = { get_global_size(0), get_global_size(1), get_global_size(2), 1 };

    // Find cube class number
    const uchar id =
        ((getValue(data, size, index.x + 0, index.y + 0, index.z + 0) > isoValue) << 0) |
        ((getValue(data, size, index.x + 1, index.y + 0, index.z + 0) > isoValue) << 1) |
        ((getValue(data, size, index.x + 1, index.y + 0, index.z + 1) > isoValue) << 2) |
        ((getValue(data, size, index.x + 0, index.y + 0, index.z + 1) > isoValue) << 3) |
        ((getValue(data, size, index.x + 0, index.y + 1, index.z + 0) > isoValue) << 4) |
        ((getValue(data, size, index.x + 1, index.y + 1, index.z + 0) > isoValue) << 5) |
        ((getValue(data, size, index.x + 1, index.y + 1, index.z + 1) > isoValue) << 6) |
        ((getValue(data, size, index.x + 0, index.y + 1, index.z + 1) > isoValue) << 7);

    // Store number of triangles
    histogram[index.x + index.y * size.x + index.z * size.x * size.y] = numberOfTriangles[id];
}
