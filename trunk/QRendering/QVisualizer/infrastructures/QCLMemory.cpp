/**
 * COPYRIGHT NOTICE
 * Copyright (c) 2012, Institute of CG & CAD, Tsinghua University.
 * All Rights Reserved.
 * 
 * @file    QCLMemory.cpp
 * @brief   QCLMemory class declaration.
 * 
 * This file declares the unified interfaces for memory processing defined in QCLMemory.h.
 * 
 * @version 1.0
 * @author  Jackie Pang
 * @e-mail: 15pengyi@gmail.com
 * @date    2012/02/07
 */

#include <gl/glew.h>
#include <cl/cl_gl.h>

#include "../utilities/QUtility.h"
#include "QCLMemory.h"

QCLMemory::QCLMemory() :
    type(QCL_UNINITIALIZED), flags(CL_MEM_READ_WRITE), bufferFormat(), imageFormat(), alwaysRead(CL_FALSE), alwaysWrite(CL_FALSE), alwaysClear(CL_FALSE), enabled(CL_TRUE), name(), size(0),
    channel(), memory(0), sampler(0), glBuffer(0), clBuffer(NULL)
{}

QCLMemory::QCLMemory(const std::string &name, const cl_mem_type &type, const cl_bool& alwaysRead, const cl_bool& alwaysWrite, const cl_mem_flags &flags, const cl_uint4& bufferFormat, const cl_bool& alwaysClear) :
    type(type), flags(flags), bufferFormat(bufferFormat), imageFormat(), alwaysRead(alwaysRead), alwaysWrite(alwaysWrite), alwaysClear(alwaysClear), enabled(CL_TRUE), name(name), size(0),
    channel(), memory(0), sampler(0), glBuffer(0), clBuffer(NULL)
{}

QCLMemory::QCLMemory(const std::string &name, const cl_mem_type &type, const cl_bool& alwaysRead, const cl_bool& alwaysWrite, const cl_mem_flags &flags, const cl_image_format& imageFormat, const cl_bool& alwaysClear) :
    type(type), flags(flags), bufferFormat(), imageFormat(imageFormat), alwaysRead(alwaysRead), alwaysWrite(alwaysWrite), alwaysClear(alwaysClear), enabled(CL_TRUE), name(name), size(0),
    channel(imageFormat), memory(0), sampler(0), glBuffer(0), clBuffer(NULL)
{}

QCLMemory::~QCLMemory()
{
    this->destroy();
}

unsigned char QCLMemory::initialize(const cl_context &context, const std::vector<::size_t> &size, void* buffer)
{
    cl_int status = CL_SUCCESS;

    this->destroy();
    this->size = size;
    this->size.resize(3, 1);
    this->clBuffer = buffer;

    switch (type)
    {
    case QCL_BUFFER:
        this->memory = clCreateBuffer(context, flags, size.at(0), NULL, &status);
        if (!QUtility::checkCLStatus(__FILE__, __LINE__, status, "clCreateBuffer()")) return GL_FALSE;
        break;
    case QCL_IMAGE2D:
        this->memory = clCreateImage2D(
            context, flags, &imageFormat,
            this->size.at(0), this->size.at(1),
            0, NULL, &status);
        if (!QUtility::checkCLStatus(__FILE__, __LINE__, status, "clCreateImage2D()")) return GL_FALSE;

        this->sampler = clCreateSampler(context, true, CL_ADDRESS_CLAMP_TO_EDGE, CL_FILTER_LINEAR, &status);
        if (!QUtility::checkCLStatus(__FILE__, __LINE__, status, "clCreateSampler()")) return GL_FALSE;
        break;
    case QCL_IMAGE3D:
        this->memory = clCreateImage3D(
            context, flags, &imageFormat, 
            this->size.at(0), this->size.at(1), this->size.at(2),
            0, 0, NULL, &status);
        if (!QUtility::checkCLStatus(__FILE__, __LINE__, status, "clCreateImage3D()")) return GL_FALSE;

        this->sampler = clCreateSampler(context, true, CL_ADDRESS_CLAMP_TO_EDGE, CL_FILTER_LINEAR, &status);
        if (!QUtility::checkCLStatus(__FILE__, __LINE__, status, "clCreateSampler()")) return GL_FALSE;
        break;
    case QCL_BUFFERGL:
        glGenBuffers(1, &glBuffer);
        if (!QUtility::checkGLStatus(__FILE__, __LINE__, "glGenBuffers()")) return GL_FALSE;
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, glBuffer);
        glBufferData(GL_PIXEL_UNPACK_BUFFER, size.at(0), 0, GL_STREAM_DRAW);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

        this->memory = clCreateFromGLBuffer(context, flags, glBuffer, &status);
        if (!QUtility::checkCLStatus(__FILE__, __LINE__, status, "clCreateFromGLBuffer()")) return GL_FALSE;
        break;
    case QCL_IMAGE2DGL:
    case QCL_IMAGE3DGL:
    default:
        this->memory = cl_mem(0);
        break;
    }

    return GL_TRUE;
}

unsigned char QCLMemory::destroy()
{
    if (this->glBuffer)
    {
        glDeleteBuffers(1, &this->glBuffer);
        if (!QUtility::checkGLStatus(__FILE__, __LINE__, "glDeleteBuffers()")) return GL_FALSE;
        this->glBuffer = GLuint(0);
    }

    if (this->memory)
    {
        cl_int status = clReleaseMemObject(this->memory);
        if (!QUtility::checkCLStatus(__FILE__, __LINE__, status, "clReleaseMemObject()")) return GL_FALSE;
        this->memory = cl_mem(0);
    }

    if (this->sampler)
    {
        cl_int status = clReleaseSampler(this->sampler);
        if (!QUtility::checkCLStatus(__FILE__, __LINE__, status, "clReleaseSampler()")) return GL_FALSE;
        this->sampler = cl_sampler(0);
    }

    return GL_TRUE;
}

unsigned char QCLMemory::read(const cl_command_queue &queue)
{
    if (!enabled) return GL_TRUE;

    cl_int status = CL_SUCCESS;

    ::size_t origin[3] = { ::size_t(0), ::size_t(0), ::size_t(0) };
    switch (type)
    {
    case QCL_BUFFER:
        status = clEnqueueWriteBuffer(queue, memory, CL_TRUE, 0, size.at(0), clBuffer, 0, NULL, NULL);
        if (!QUtility::checkCLStatus(__FILE__, __LINE__, status, "clEnqueueWriteBuffer()")) return GL_FALSE;
        break;
    case QCL_IMAGE2D:
        status = clEnqueueWriteImage(
            queue, memory, CL_TRUE,
            origin, size.data(),
            size.at(0) * channel.number * channel.size, 0,
            clBuffer, 0, NULL, NULL);
        if (!QUtility::checkCLStatus(__FILE__, __LINE__, status, "clEnqueueWriteImage()")) return GL_FALSE;
        break;
    case QCL_IMAGE3D:
        status = clEnqueueWriteImage(
            queue, memory, CL_TRUE,
            origin, size.data(),
            size.at(0) * channel.number * channel.size, size.at(0) * size.at(1) * channel.number * channel.size,
            clBuffer, 0, NULL, NULL);
        if (!QUtility::checkCLStatus(__FILE__, __LINE__, status, "clEnqueueWriteImage()")) return GL_FALSE;
        break;
    case QCL_BUFFERGL:
        status = clEnqueueAcquireGLObjects(queue, 1, &memory, 0, NULL, NULL);
        if (!QUtility::checkCLStatus(__FILE__, __LINE__, status, "clEnqueueAcquireGLObjects()")) return GL_FALSE;
        break;
    case QCL_IMAGE2DGL:
    case QCL_IMAGE3DGL:
    default:
        break;
    }

    return GL_TRUE;
}

unsigned char QCLMemory::read(const cl_command_queue &queue, const std::vector<::size_t> &bufferOrigin, const std::vector<::size_t> &hostOrigin,
    const std::vector<::size_t> &size, const std::vector<::size_t> &pitch)
{
    if (!enabled) return GL_TRUE;

    cl_int status = CL_SUCCESS;

    switch (type)
    {
    case QCL_BUFFER:
        status = clEnqueueWriteBufferRect(queue, memory, CL_TRUE,
            bufferOrigin.data(), hostOrigin.data(), size.data(),
            pitch.at(0), pitch.at(1),
            pitch.at(0), pitch.at(1),
            clBuffer, 0, NULL, NULL);
        if (!QUtility::checkCLStatus(__FILE__, __LINE__, status, "clEnqueueWriteBufferRect()")) return GL_FALSE;
        break;
    case QCL_IMAGE2D:
    case QCL_IMAGE3D:
    case QCL_IMAGE2DGL:
    case QCL_IMAGE3DGL:
    default:
        break;
    }

    return GL_TRUE;
}

unsigned char QCLMemory::write(const cl_command_queue &queue)
{
    if (!enabled) return GL_TRUE;

    cl_int status = CL_SUCCESS;

    switch (type)
    {
    case QCL_BUFFER:
        status = clEnqueueReadBuffer(queue, memory, CL_TRUE, 0, size.at(0), clBuffer, 0, NULL, NULL);
        if (!QUtility::checkCLStatus(__FILE__, __LINE__, status, "clEnqueueReadBuffer()")) return GL_FALSE;
        break;
    case QCL_IMAGE2D:
    case QCL_IMAGE3D:
        status = clEnqueueReadImage(
            queue, memory, CL_TRUE,
            NULL, size.data(),
            size.at(0) * channel.number * channel.size, size.at(0) * size.at(1) * channel.number * channel.size,
            clBuffer, 0, NULL, NULL);
        if (!QUtility::checkCLStatus(__FILE__, __LINE__, status, "clEnqueueReadImage()")) return GL_FALSE;
        break;
    case QCL_BUFFERGL:
        status = clEnqueueReleaseGLObjects(queue, 1, &memory, 0, NULL, NULL);
        if (!QUtility::checkCLStatus(__FILE__, __LINE__, status, "clEnqueueReleaseGLObjects()")) return GL_FALSE;
        break;
    case QCL_IMAGE2DGL:
    case QCL_IMAGE3DGL:
    default:
        break;
    }

    return GL_TRUE;
}

unsigned char QCLMemory::write(const cl_command_queue &queue, const std::vector<::size_t> &bufferOrigin, const std::vector<::size_t> &hostOrigin,
    const std::vector<::size_t> &size, const std::vector<::size_t> &pitch)
{
    if (!enabled) return GL_TRUE;

    cl_int status = CL_SUCCESS;

    ::size_t origin[3] = { ::size_t(0), ::size_t(0), ::size_t(0) };
    switch (type)
    {
    case QCL_BUFFER:
        status = clEnqueueReadBufferRect(queue, memory, CL_TRUE,
            bufferOrigin.data(), hostOrigin.data(), size.data(),
            pitch.at(0), pitch.at(1),
            pitch.at(0), pitch.at(1),
            clBuffer, 0, NULL, NULL);
        if (!QUtility::checkCLStatus(__FILE__, __LINE__, status, "clEnqueueReadBufferRect()")) return GL_FALSE;
        break;
    case QCL_IMAGE2D:
    case QCL_IMAGE3D:
    case QCL_IMAGE2DGL:
    case QCL_IMAGE3DGL:
    default:
        break;
    }

    return GL_TRUE;
}

const cl_mem& QCLMemory::get()
{
    return memory;
}

const cl_sampler& QCLMemory::getSampler()
{
    return sampler;
}

const cl_uint QCLMemory::getSize()
{
    switch (type)
    {
    case QCL_BUFFER:
        return bufferFormat.s[0] * bufferFormat.s[1] * bufferFormat.s[2] * bufferFormat.s[3];
    case QCL_IMAGE2D:
    case QCL_IMAGE3D:
        {
            cl_uint s = channel.number * channel.size;
            for (std::vector<::size_t>::iterator i = size.begin(); i != size.end(); i++)
                s *= *i;
            return s;
        }
    case QCL_BUFFERGL:
    case QCL_IMAGE2DGL:
    case QCL_IMAGE3DGL:
    default:
        return 0;
    }
}

const void* QCLMemory::getBuffer()
{
    switch (type)
    {
    case QCL_BUFFER:
    case QCL_IMAGE2D:
    case QCL_IMAGE3D:
        return clBuffer;
    case QCL_BUFFERGL:
        return &glBuffer;
    case QCL_IMAGE2DGL:
    case QCL_IMAGE3DGL:
    default:
        return NULL;
    }
}

void QCLMemory::setBuffer(void* buffer)
{
    switch (type)
    {
    case QCL_BUFFER:
    case QCL_IMAGE2D:
    case QCL_IMAGE3D:
        clBuffer = buffer;
        break;
    case QCL_BUFFERGL:
    case QCL_IMAGE2DGL:
    case QCL_IMAGE3DGL:
        break;
    default:
        break;
    }
}

std::list<QCLMemory>::iterator QCLMemory::find(std::list<QCLMemory> &memories, const std::string &name)
{
    for (std::list<QCLMemory>::iterator i = memories.begin(); i != memories.end(); i++)
    {
        if (i->name.find(name) != std::string::npos) return i;
    }
    return memories.end();
}