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

__kernel void viewInit(
        const uint histogramSize,
        __global ulong *bufferData
         )
{
    uint x = get_global_id(0);
    uint y = get_global_id(1);
    uint z = get_global_id(2);
    __global ulong *data = bufferData + (x + y * get_global_size(0)) * histogramSize + z;
    data[0] = 0;
}