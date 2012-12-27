/**
 * COPYRIGHT NOTICE
 * Copyright (c) 2012, Institute of CG & CAD, Tsinghua University.
 * All Rights Reserved.
 * 
 * @file    QUtility.cpp
 * @brief   QUtility class declaration.
 * 
 * This file declares the most commonly used methods defined in QUtility.h.
 * 
 * @version 1.0
 * @author  Jackie Pang
 * @e-mail: 15pengyi@gmail.com
 * @date    2012/02/07
 */

#include <gl/glew.h>
#include <cl/cl_gl.h>

#include <iostream>
#include <fstream>

#include <QTime>

#include "../infrastructures/QStructure.h"
#include "QUtility.h"

QUtility::QUtility()
{}

QUtility::~QUtility()
{}

::size_t QUtility::roundUp(::size_t group_size, ::size_t global_size) 
{
    ::size_t r = global_size % group_size;
    return r == 0 ? global_size : global_size + group_size - r;
}

void QUtility::printTimeCost(unsigned long milliseconds, const std::string& name)
{
    std::cerr << " > LOG: " << name.c_str() << " " << milliseconds << " ms." << std::endl;
}

void QUtility::printFPS(unsigned long milliseconds, const std::string& name)
{
    double fps = milliseconds == 0 ? 1000.0 : 1000.0 / milliseconds;
    std::cerr << " > LOG: " << name.c_str() << " " << fps << " fps." << std::endl;
}

void QUtility::printBandWidth(::size_t bytes, unsigned long milliseconds, const std::string& name)
{
    double rate = milliseconds == 0 ? 1000.0 : 1000.0 * bytes / (1 << 20) / milliseconds;
    std::cerr << " > LOG: " << name.c_str() << " " << rate << " MB/s." << std::endl;
}

unsigned char  QUtility::checkGLStatus(char *file, int line, char *name)
{
    GLenum error = glGetError();
    while (error != GL_NO_ERROR)
    {
        std::cerr << " > ERROR: " << file << "(" << line << ") " << name << " failed. ";
        switch (error)
        {
        case GL_INVALID_ENUM:
            std::cerr << "an unacceptable value is specified for an enumerated argument. The offending command is ignored, and has no other side effect than to set the error flag.";
            break;
        case GL_INVALID_VALUE:
            std::cerr << "a numeric argument is out of range. The offending command is ignored, and has no other side effect than to set the error flag.";
            break;
        case GL_STACK_OVERFLOW:
            std::cerr << "this command would cause a stack overflow. The offending command is ignored, and has no other side effect than to set the error flag.";
            break;
        case GL_STACK_UNDERFLOW:
            std::cerr << "this command would cause a stack underflow. The offending command is ignored, and has no other side effect than to set the error flag.";
            break;
        case GL_OUT_OF_MEMORY:
            std::cerr << "there is not enough memory left to execute the command. The state of the GL is undefined, except for the state of the error flags, after this error is recorded.";
            break;
        case GL_INVALID_OPERATION:
            std::cerr << "the specified operation is not allowed in the current state. The offending command is ignored, and has no other side effect than to set the error flag.";
            break;
        default:
            std::cerr << "unknown.";
            break;
        }
        std::cerr << std::endl;

        if (error = glGetError() == GL_NO_ERROR) return GL_FALSE;
    }
    return GL_TRUE;
}

unsigned char QUtility::checkCLStatus(char *file, int line, cl_int status, char *name)
{
    if (status != CL_SUCCESS)
    {
        std::cerr << " > ERROR: " << file << "(" << line << ") " << name << " failed (";
        switch (status)
        {
        case CL_DEVICE_NOT_FOUND:
            std::cerr << status << ", CL_DEVICE_NOT_FOUND";
            break;
        case CL_DEVICE_NOT_AVAILABLE:
            std::cerr << status << ", CL_DEVICE_NOT_AVAILABLE";
            break;
        case CL_COMPILER_NOT_AVAILABLE:
            std::cerr << status << ", CL_COMPILER_NOT_AVAILABLE";
            break;
        case CL_MEM_OBJECT_ALLOCATION_FAILURE:
            std::cerr << status << ", CL_MEM_OBJECT_ALLOCATION_FAILURE";
            break;
        case CL_OUT_OF_RESOURCES:
            std::cerr << status << ", CL_OUT_OF_RESOURCES";
            break;
        case CL_OUT_OF_HOST_MEMORY:
            std::cerr << status << ", CL_OUT_OF_HOST_MEMORY";
            break;
        case CL_PROFILING_INFO_NOT_AVAILABLE:
            std::cerr << status << ", CL_PROFILING_INFO_NOT_AVAILABLE";
            break;
        case CL_MEM_COPY_OVERLAP:
            std::cerr << status << ", CL_MEM_COPY_OVERLAP";
            break;
        case CL_IMAGE_FORMAT_MISMATCH:
            std::cerr << status << ", CL_IMAGE_FORMAT_MISMATCH";
            break;
        case CL_IMAGE_FORMAT_NOT_SUPPORTED:
            std::cerr << status << ", CL_IMAGE_FORMAT_NOT_SUPPORTED";
            break;
        case CL_BUILD_PROGRAM_FAILURE:
            std::cerr << status << ", CL_BUILD_PROGRAM_FAILURE";
            break;
        case CL_MAP_FAILURE:
            std::cerr << status << ", CL_MAP_FAILURE";
            break;
            /*
        case CL_MISALIGNED_SUB_BUFFER_OFFSET:
            std::cerr << status << ", CL_MISALIGNED_SUB_BUFFER_OFFSET";
            break;
        case CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST:
            std::cerr << status << ", CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
            break;
            */
        case CL_INVALID_VALUE:
            std::cerr << status << ", CL_INVALID_VALUE";
            break;
        case CL_INVALID_DEVICE_TYPE:
            std::cerr << status << ", CL_INVALID_DEVICE_TYPE";
            break;
        case CL_INVALID_PLATFORM:
            std::cerr << status << ", CL_INVALID_PLATFORM";
            break;
        case CL_INVALID_DEVICE:
            std::cerr << status << ", CL_INVALID_DEVICE";
            break;
        case CL_INVALID_CONTEXT:
            std::cerr << status << ", CL_INVALID_CONTEXT";
            break;
        case CL_INVALID_QUEUE_PROPERTIES:
            std::cerr << status << ", CL_INVALID_QUEUE_PROPERTIES";
            break;
        case CL_INVALID_COMMAND_QUEUE:
            std::cerr << status << ", CL_INVALID_COMMAND_QUEUE";
            break;
        case CL_INVALID_HOST_PTR:
            std::cerr << status << ", CL_INVALID_HOST_PTR";
            break;
        case CL_INVALID_MEM_OBJECT:
            std::cerr << status << ", CL_INVALID_MEM_OBJECT";
            break;
        case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:
            std::cerr << status << ", CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
            break;
        case CL_INVALID_IMAGE_SIZE:
            std::cerr << status << ", CL_INVALID_IMAGE_SIZE";
            break;
        case CL_INVALID_SAMPLER:
            std::cerr << status << ", CL_INVALID_SAMPLER";
            break;
        case CL_INVALID_BINARY:
            std::cerr << status << ", CL_INVALID_BINARY";
            break;
        case CL_INVALID_BUILD_OPTIONS:
            std::cerr << status << ", CL_INVALID_BUILD_OPTIONS";
            break;
        case CL_INVALID_PROGRAM:
            std::cerr << status << ", CL_INVALID_PROGRAM";
            break;
        case CL_INVALID_PROGRAM_EXECUTABLE:
            std::cerr << status << ", CL_INVALID_PROGRAM_EXECUTABLE";
            break;
        case CL_INVALID_KERNEL_NAME:
            std::cerr << status << ", CL_INVALID_KERNEL_NAME";
            break;
        case CL_INVALID_KERNEL_DEFINITION:
            std::cerr << status << ", CL_INVALID_KERNEL_DEFINITION";
            break;
        case CL_INVALID_KERNEL:
            std::cerr << status << ", CL_INVALID_KERNEL";
            break;
        case CL_INVALID_ARG_INDEX:
            std::cerr << status << ", CL_INVALID_ARG_INDEX";
            break;
        case CL_INVALID_ARG_VALUE:
            std::cerr << status << ", CL_INVALID_ARG_VALUE";
            break;
        case CL_INVALID_ARG_SIZE:
            std::cerr << status << ", CL_INVALID_ARG_SIZE";
            break;
        case CL_INVALID_KERNEL_ARGS:
            std::cerr << status << ", CL_INVALID_KERNEL_ARGS";
            break;
        case CL_INVALID_WORK_DIMENSION:
            std::cerr << status << ", CL_INVALID_WORK_DIMENSION";
            break;
        case CL_INVALID_WORK_GROUP_SIZE:
            std::cerr << status << ", CL_INVALID_WORK_GROUP_SIZE";
            break;
        case CL_INVALID_WORK_ITEM_SIZE:
            std::cerr << status << ", CL_INVALID_WORK_ITEM_SIZE";
            break;
        case CL_INVALID_GLOBAL_OFFSET:
            std::cerr << status << ", CL_INVALID_GLOBAL_OFFSET";
            break;
        case CL_INVALID_EVENT_WAIT_LIST:
            std::cerr << status << ", CL_INVALID_EVENT_WAIT_LIST";
            break;
        case CL_INVALID_EVENT:
            std::cerr << status << ", CL_INVALID_EVENT";
            break;
        case CL_INVALID_OPERATION:
            std::cerr << status << ", CL_INVALID_OPERATION";
            break;
        case CL_INVALID_GL_OBJECT:
            std::cerr << status << ", CL_INVALID_GL_OBJECT";
            break;
        case CL_INVALID_BUFFER_SIZE:
            std::cerr << status << ", CL_INVALID_BUFFER_SIZE";
            break;
        case CL_INVALID_MIP_LEVEL:
            std::cerr << status << ", CL_INVALID_MIP_LEVEL";
            break;
        case CL_INVALID_GLOBAL_WORK_SIZE:
            std::cerr << status << ", CL_INVALID_GLOBAL_WORK_SIZE";
            break;
        case CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR:
            std::cerr << status << ", CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR";
        }
        std::cerr << ")" << std::endl;
        return GL_FALSE;
    }
    return GL_TRUE;
}

unsigned char QUtility::checkShaderStatus(char *file, int line, GLuint shader, GLenum name)
{
    GLint status = GL_TRUE;
    glGetShaderiv(shader, name, &status);
    if (!status)
    {
        std::cerr << " > ERROR: " << file << "(" << line << ") checkShaderStatus()." << std::endl;
        return GL_FALSE;
    }

    return GL_TRUE;
}

unsigned char QUtility::checkProgramStatus(char *file, int line, GLuint program, GLenum name)
{
    GLint status = GL_TRUE;
    glGetProgramiv(program, name, &status);
    if (!status)
    {
        std::cerr << " > ERROR: " << file << "(" << line << ") checkProgramStatus()." << std::endl;
        return GL_FALSE;
    }

    return GL_TRUE;
}

unsigned char QUtility::checkBufferStatus(int number)
{
    int maxBuffers = 0;
    glGetIntegerv(GL_MAX_DRAW_BUFFERS, &maxBuffers);
    if (maxBuffers < number)
    {
        std::cerr << " > ERROR: checking buffers." << std::endl;
        return GL_FALSE;
    }
    return GL_TRUE;
}

unsigned char QUtility::checkFramebufferStatus(unsigned int target)
{
    unsigned int status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
    if (status != GL_FRAMEBUFFER_COMPLETE)
    {
        std::cerr << " > ERROR: ";
        switch (status)
        {
        case GL_FRAMEBUFFER_UNDEFINED:
            std::cerr << "target is the default framebuffer, but the default framebuffer does not exist.";
            break;
        case GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT:
            std::cerr << "any of the framebuffer attachment points are framebuffer incomplete.";
            break;
        case GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT:
            std::cerr << "the framebuffer does not have at least one image attached to it.";
            break;
        case GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER:
            std::cerr << "the value of GL_FRAMEBUFFER_ATTACHMENT_OBJECT_TYPE is GL_NONE for any color attachment point(s) named by GL_DRAWBUFFERi.";
            break;
        case GL_FRAMEBUFFER_INCOMPLETE_READ_BUFFER:
            std::cerr << "GL_READ_BUFFER is not GL_NONE and the value of GL_FRAMEBUFFER_ATTACHMENT_OBJECT_TYPE is GL_NONE for the color attachment point named by GL_READ_BUFFER.";
            break;
        case GL_FRAMEBUFFER_UNSUPPORTED:
            std::cerr << "the combination of internal formats of the attached images violates an implementation-dependent set of restrictions";
            break;
        case GL_FRAMEBUFFER_INCOMPLETE_MULTISAMPLE:
            std::cerr << "the value of GL_RENDERBUFFER_SAMPLES is not the same for all attached renderbuffers; the value of GL_TEXTURE_SAMPLES is the not same for all attached textures; or, the attached images are a mix of renderbuffers and textures, the value of GL_RENDERBUFFER_SAMPLES does not match the value of GL_TEXTURE_SAMPLES; the value of GL_TEXTURE_FIXED_SAMPLE_LOCATIONS is not the same for all attached textures; or, the attached images are a mix of renderbuffers and textures, the value of GL_TEXTURE_FIXED_SAMPLE_LOCATIONS is not GL_TRUE for all attached textures.";
            break;
        case GL_FRAMEBUFFER_INCOMPLETE_LAYER_TARGETS:
            std::cerr << "any framebuffer attachment is layered, and any populated attachment is not layered, or all populated color attachments are not from textures of the same target.";
            break;
        default:
            std::cerr << "unknown.";
            break;
        }
        std::cerr << std::endl;
        return GL_FALSE;
    }
    return GL_TRUE;
}

unsigned char QUtility::checkSupport()
{
    return QUtility::checkOpenglSupport() && QUtility::checkGlewSupport() && QUtility::checkShaderSupport();
}

unsigned char QUtility::checkGlewSupport()
{
    unsigned int error = glewInit();
    if (error != GLEW_OK || !glewIsSupported("GL_VERSION_2_1"))
    {
        std::cerr << " > ERROR: checking glew support." << std::endl;
        std::cerr << " > ERROR: " << glewGetErrorString(error) << std::endl;
        return GL_FALSE;
    }
    return GL_TRUE;
}

unsigned char QUtility::checkOpenglSupport()
{
    if (!GL_VERSION_3_0)
    {
        std::cerr << " > ERROR: checking opengl support." << std::endl;
        return GL_FALSE;
    }
    else
    {
        std::cerr << " > GL_VERSION: " << glGetString(GL_VERSION) << std::endl;
    }
    return GL_TRUE;
}

unsigned char QUtility::checkShaderSupport()
{
    if (!GL_VERTEX_SHADER || !GL_FRAGMENT_SHADER)
    {
        std::cerr << " > ERROR: checking shader support." << std::endl;
        return GL_FALSE;
    }
    return GL_TRUE;
}

unsigned char QUtility::checkArguments(int argc, char *argv[])
{
    if (argc != 2)
    {
        std::cerr << " > ERROR: checking arguments." << std::endl;
        std::cerr << " > Usage: jrendering <file.dat>." << std::endl;
        return GL_FALSE;
    }
    return GL_TRUE;
}

void QUtility::trim(std::string &s)
{
    if (!s.empty())
    {
        int found = s.find_first_of('\t');
        while (found != std::string::npos)
        {
            s.replace(found, 1, " ");
            found = s.find_first_of('\t', found + 1);
        }
        s.erase(0, s.find_first_not_of(' '));
        s.erase(s.find_last_not_of(' ') + 1);
    }
}

double QUtility::smooth(int edge0, int edge1, int x)
{
    if (x <= edge0)
        return 0.0;
    else if (x >= edge1)
        return 1.0;
    else
        return (double)(x - edge0) / (edge1 - edge0);
}

unsigned int QUtility::getSize(unsigned int target)
{
    switch (target)
    {
    case GL_UNSIGNED_BYTE:
        return 1;
    case GL_UNSIGNED_SHORT:
        return 2;
    case GL_FLOAT:
        return 4;
    case GL_DOUBLE:
        return 8;
    default:
        return 0;
    }
}

void QUtility::preprocess(void* volumeData, const ::size_t& volumeSize, const QDataFormat& format,
    const QEndianness& endian, const ::size_t& histogramSize, float* histogramData,
    float& volumeMin, float& volumeMax)
{
    float histogramMin(0.0f), histogramMax(0.0f);
    preprocess(volumeData, volumeSize, format, endian, histogramSize, histogramData, volumeMin, volumeMax, histogramMin, histogramMax);
}

void QUtility::preprocess(void* volumeData, const ::size_t& volumeSize, const QDataFormat& format,
    const QEndianness& endian, const ::size_t& histogramSize, float* histogramData,
    float& volumeMin, float& volumeMax, float& histogramMin, float& histogramMax)
{
    switch (format)
    {
    case DATA_UCHAR:
        QUtilityTemplate<unsigned char>::normalize(volumeData, volumeSize, endian, volumeMin, volumeMax);
        break;
    case DATA_USHORT:
        QUtilityTemplate<unsigned short>::normalize(volumeData, volumeSize, endian, volumeMin, volumeMax);
        break;
    case DATA_FLOAT:
        QUtilityTemplate<float>::normalize(volumeData, volumeSize, endian, volumeMin, volumeMax);
        break;
    default:
        break;
    }

    if (histogramSize > 0)
    {
        float level = histogramSize - 1;
        float* volumeBegin = (float*)volumeData;
        float* volumeEnd = volumeBegin + volumeSize;
        for (float *i = volumeBegin; i != volumeEnd; i++)
            histogramData[(int)(*i * level)]++;

        QUtilityTemplate<float>::computeLogarithm(histogramSize, histogramData, histogramMin, histogramMax);
    }
}