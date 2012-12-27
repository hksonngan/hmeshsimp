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

void addValue8(
    __global uchar* data,
    __private uint4 size,
    __private uint x,
    __private uint y,
    __private uint z,
    __private uint* result
    )
{
    if (x < size.x && y < size.y && z < size.z)
        *result += data[x + y * size.x + z * size.x * size.y];
}

void addValue16(
    __global ushort* data,
    __private uint4 size,
    __private uint x,
    __private uint y,
    __private uint z,
    __private uint* result
    )
{
    if (x < size.x && y < size.y && z < size.z)
        *result += data[x + y * size.x + z * size.x * size.y];
}

void addValue32(
    __global uint* data,
    __private uint4 size,
    __private uint x,
    __private uint y,
    __private uint z,
    __private uint* result
    )
{
    if (x < size.x && y < size.y && z < size.z)
        *result += data[x + y * size.x + z * size.x * size.y];
}

__kernel void constructHP(
    __private uint4 readFormat,
    __global uchar* readHistogram,
    __private uint4 writeFormat,
    __global uchar* writeHistogram
    )
{
    uint4 writeIndex  = { get_global_id(0), get_global_id(1), get_global_id(2), 0 };
    uint  writeOffset = writeIndex.x + writeIndex.y * writeFormat.x + writeIndex.z * writeFormat.x * writeFormat.y;

    uint4 readIndex  = writeIndex * 2;
    uint  readOffset = readIndex.x + readIndex.y * readFormat.x + readIndex.z * readFormat.x * readFormat.y;
    
    uint result = 0;
    if (readFormat.w == 1)
    {
        __global uchar* readData = (__global uchar*)readHistogram;
        addValue8(readData, readFormat, readIndex.x + 0, readIndex.y + 0, readIndex.z + 0, &result);
        addValue8(readData, readFormat, readIndex.x + 1, readIndex.y + 0, readIndex.z + 0, &result);
        addValue8(readData, readFormat, readIndex.x + 0, readIndex.y + 1, readIndex.z + 0, &result);
        addValue8(readData, readFormat, readIndex.x + 1, readIndex.y + 1, readIndex.z + 0, &result);
        addValue8(readData, readFormat, readIndex.x + 0, readIndex.y + 0, readIndex.z + 1, &result);
        addValue8(readData, readFormat, readIndex.x + 1, readIndex.y + 0, readIndex.z + 1, &result);
        addValue8(readData, readFormat, readIndex.x + 0, readIndex.y + 1, readIndex.z + 1, &result);
        addValue8(readData, readFormat, readIndex.x + 1, readIndex.y + 1, readIndex.z + 1, &result);
    }
    else if (readFormat.w == 2)
    {
        __global ushort* readData = (__global ushort*)readHistogram;
        addValue16(readData, readFormat, readIndex.x + 0, readIndex.y + 0, readIndex.z + 0, &result);
        addValue16(readData, readFormat, readIndex.x + 1, readIndex.y + 0, readIndex.z + 0, &result);
        addValue16(readData, readFormat, readIndex.x + 0, readIndex.y + 1, readIndex.z + 0, &result);
        addValue16(readData, readFormat, readIndex.x + 1, readIndex.y + 1, readIndex.z + 0, &result);
        addValue16(readData, readFormat, readIndex.x + 0, readIndex.y + 0, readIndex.z + 1, &result);
        addValue16(readData, readFormat, readIndex.x + 1, readIndex.y + 0, readIndex.z + 1, &result);
        addValue16(readData, readFormat, readIndex.x + 0, readIndex.y + 1, readIndex.z + 1, &result);
        addValue16(readData, readFormat, readIndex.x + 1, readIndex.y + 1, readIndex.z + 1, &result);
    }
    else
    {
        __global uint* readData = (__global uint*)readHistogram;
        addValue32(readData, readFormat, readIndex.x + 0, readIndex.y + 0, readIndex.z + 0, &result);
        addValue32(readData, readFormat, readIndex.x + 1, readIndex.y + 0, readIndex.z + 0, &result);
        addValue32(readData, readFormat, readIndex.x + 0, readIndex.y + 1, readIndex.z + 0, &result);
        addValue32(readData, readFormat, readIndex.x + 1, readIndex.y + 1, readIndex.z + 0, &result);
        addValue32(readData, readFormat, readIndex.x + 0, readIndex.y + 0, readIndex.z + 1, &result);
        addValue32(readData, readFormat, readIndex.x + 1, readIndex.y + 0, readIndex.z + 1, &result);
        addValue32(readData, readFormat, readIndex.x + 0, readIndex.y + 1, readIndex.z + 1, &result);
        addValue32(readData, readFormat, readIndex.x + 1, readIndex.y + 1, readIndex.z + 1, &result);
    }
    
    if (writeFormat.w == 1)
    {
        __global uchar* writeData = (__global uchar*)writeHistogram + writeOffset;
        writeData[0] = result;
    }
    else if (writeFormat.w == 2)
    {
        __global ushort* writeData = (__global ushort*)writeHistogram + writeOffset;
        writeData[0] = result;
    }
    else
    {
        __global uint* writeData = (__global uint*)writeHistogram + writeOffset;
        writeData[0] = result;
    }
}
