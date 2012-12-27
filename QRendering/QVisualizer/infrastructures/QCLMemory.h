/**
 * COPYRIGHT NOTICE
 * Copyright (c) 2012, Institute of CG & CAD, Tsinghua University.
 * All Rights Reserved.
 * 
 * @file    QCLMemory.h
 * @brief   QCLMemory class definition.
 * 
 * This file wraps the memory processing functions in OpenCL to provide users unified interfaces of
 *     memory initialization, destory, reading(from CPU to GPU) and writing(from GPU to CPU).
 * 
 * @version 1.0
 * @author  Jackie Pang
 * @e-mail: 15pengyi@gmail.com
 * @date    2012/02/07
 */

#ifndef QCLMEMORY_H
#define QCLMEMORY_H

#include <string>
#include <list>
#include <vector>

struct cl_channel
{
    cl_channel() :
        number(0), size(0)
    {}

    cl_channel(const cl_image_format& format)
    {
        switch (format.image_channel_order)
        {
        case CL_R:
        case CL_Rx:
        case CL_A:
        case CL_INTENSITY:
        case CL_LUMINANCE:
            this->number = cl_uint(1);
            break;
        case CL_RG:
        case CL_RGx:
        case CL_RA:
            this->number = cl_uint(2);
            break;
        case CL_RGB:
        case CL_RGBx:
            this->number = cl_uint(3);
            break;
        case CL_RGBA:
        case CL_BGRA:
        case CL_ARGB:
            this->number = cl_uint(4);
            break;
        default:
            this->number = cl_uint(0);
            break;
        }

        switch (format.image_channel_data_type)
        {
        case CL_SNORM_INT8:
        case CL_UNORM_INT8:
        case CL_SIGNED_INT8:
        case CL_UNSIGNED_INT8:
        case CL_LUMINANCE:
            this->size = cl_uint(1);
            break;
        case CL_SNORM_INT16:
        case CL_UNORM_INT16:
        case CL_UNORM_SHORT_565:
        case CL_UNORM_SHORT_555:
        case CL_SIGNED_INT16:
        case CL_UNSIGNED_INT16:
        case CL_HALF_FLOAT:
            this->size = cl_uint(2);
            break;
        case CL_UNORM_INT_101010:
        case CL_SIGNED_INT32:
        case CL_UNSIGNED_INT32:
        case CL_FLOAT:
            this->size = cl_uint(4);
            break;
        default:
            this->size = cl_uint(0);
            break;
        }
    }

    ~cl_channel() {}

    cl_uint number;
    cl_uint size;
};

class QCLMemory
{
public:
    enum cl_mem_type
    {
        QCL_UNINITIALIZED   = 0,
        QCL_BUFFER          = 1, 
        QCL_IMAGE2D         = 2,
        QCL_IMAGE3D         = 3,
        QCL_BUFFERGL        = 4,
        QCL_IMAGE2DGL       = 5,
        QCL_IMAGE3DGL       = 6
    };
    
    QCLMemory();
    QCLMemory(const std::string &name, const cl_mem_type &type, const cl_bool& alwaysRead, const cl_bool& alwaysWrite, const cl_mem_flags &flags, const cl_uint4& bufferFormat = cl_uint4(), const cl_bool& alwaysClear = CL_FALSE);
    QCLMemory(const std::string &name, const cl_mem_type &type, const cl_bool& alwaysRead, const cl_bool& alwaysWrite, const cl_mem_flags &flags, const cl_image_format& imageFormat, const cl_bool& alwaysClear = CL_FALSE);
    ~QCLMemory();
    
    const cl_mem_type type;
    const cl_bool alwaysRead, alwaysWrite, alwaysClear;
    const cl_mem_flags flags;
    const cl_image_format imageFormat;
    const cl_uint4 bufferFormat;
    const std::string name;

    cl_bool enabled;
    std::vector<::size_t> size;
    
    unsigned char initialize(const cl_context &context, const std::vector<::size_t> &size, void* buffer = NULL);
    unsigned char destroy();
    unsigned char read(const cl_command_queue &queue);
    unsigned char read(const cl_command_queue &queue, const std::vector<::size_t> &bufferOrigin, const std::vector<::size_t> &hostOrigin,
        const std::vector<::size_t> &size, const std::vector<::size_t> &pitch);
    unsigned char write(const cl_command_queue &queue);
    unsigned char write(const cl_command_queue &queue, const std::vector<::size_t> &bufferOrigin, const std::vector<::size_t> &hostOrigin,
        const std::vector<::size_t> &size, const std::vector<::size_t> &pitch);
    const cl_mem& get();
    const cl_sampler& getSampler();
    const cl_uint QCLMemory::getSize();
    const void* getBuffer();
    void setBuffer(void* buffer = NULL);
    
    static std::list<QCLMemory>::iterator find(std::list<QCLMemory> &memories, const std::string &name);
    
private:
    const cl_channel channel;

    cl_mem memory;
    cl_sampler sampler;
    cl_uint glBuffer;
    void* clBuffer;
};

#endif  // QCLMEMORY_H