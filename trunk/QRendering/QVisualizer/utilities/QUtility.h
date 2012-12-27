/**
 * COPYRIGHT NOTICE
 * Copyright (c) 2012, Institute of CG & CAD, Tsinghua University.
 * All Rights Reserved.
 * 
 * @file    QUtility.h
 * @brief   QUtility class definition.
 * 
 * This file defines a utility class which contains serveral commonly used methods.
 * These methods can be devided into three groups:
 *     a group for checking errors, 
 *     a group for printing debug information,
 *     and a group for data preprocessing.
 * 
 * @version 1.0
 * @author  Jackie Pang
 * @e-mail: 15pengyi@gmail.com
 * @date    2012/02/07
 */

#ifndef QUTILITY_H
#define QUTILITY_H

#include <gl/glew.h>
#include <cl/cl.h>

#include <vector>
#include <iostream>

#include "../infrastructures/QStructure.h"

// [houtao]
#include "float.h"
#include "math.h"

#if defined (__APPLE__) || defined(MACOSX)
#define GL_SHARING_EXTENSION "cl_APPLE_gl_sharing"
#else
#define GL_SHARING_EXTENSION "cl_khr_gl_sharing"
#endif

enum QEndianness;

class QUtility
{
public:
    QUtility();
    ~QUtility();
    
    static ::size_t roundUp(::size_t group_size, ::size_t global_size);
    static void printFPS(unsigned long milliseconds, const std::string& name = std::string());
    static void printTimeCost(unsigned long milliseconds, const std::string& name = std::string());
    static void printBandWidth(::size_t bytes, unsigned long milliseconds, const std::string& name = std::string());
    static void trim(std::string &s);
    static double smooth(int edge0, int edge1, int x);
    static unsigned char checkSupport();
    static unsigned char checkGlewSupport();
    static unsigned char checkOpenglSupport();
    static unsigned char checkGLStatus(char *file, int line, char *name);
    static unsigned char checkCLStatus(char *file, int line, cl_int status, char *name);
    static unsigned char checkShaderStatus(char *file, int line, unsigned int shader, GLenum name);
    static unsigned char checkProgramStatus(char *file, int line, unsigned int program, GLenum name);
    static unsigned char checkBufferStatus(int number);
    static unsigned char checkFramebufferStatus(unsigned int target);
    static unsigned char checkShaderSupport();
    static unsigned char checkArguments(int argc, char *argv[]);
    static unsigned int getSize(unsigned int target);

    static void preprocess(void* volumeData, const ::size_t& volumeSize, const QDataFormat& format,
        const QEndianness& endian, const ::size_t& histogramSize, float* histogramData,
        float& volumeMin, float& volumeMax);
    static void preprocess(void* volumeData, const ::size_t& volumeSize, const QDataFormat& format,
        const QEndianness& endian, const ::size_t& histogramSize, float* histogramData,
        float& volumeMin, float& volumeMax, float& histogramMin, float& histogramMax);
};

struct TypeFalse
{
    enum { value = false };
};

struct TypeTrue
{
    enum { value = true };
};

template <typename T1, typename T2>
struct TypeIsSame
{
    typedef TypeFalse result;
};

template <typename T>
struct TypeIsSame<T,T>
{
    typedef TypeTrue result;
};

template <typename T>
class QUtilityTemplate
{
public:
    QUtilityTemplate();
    ~QUtilityTemplate();

    static void read(const std::string &content, int &count, void* start, const ::size_t& size)
    {
        char* buffer((char*)content.data() + count);
        T* end = (T*)start + size;
        if (TypeIsSame<T, char>::result::value || TypeIsSame<T, short>::result::value || TypeIsSame<T, int>::result::value || TypeIsSame<T, long>::result::value)
        {
            for (T* i = (T*)start; i != end; i++)
                *i = strtol(buffer, &buffer, 10);
        }
        else if (TypeIsSame<T, unsigned char>::result::value || TypeIsSame<T, unsigned short>::result::value || TypeIsSame<T, unsigned int>::result::value || TypeIsSame<T, unsigned long>::result::value)
        {
            for (T* i = (T*)start; i != end; i++)
                *i = strtoul(buffer, &buffer, 10);
        }
        else if (TypeIsSame<T, float>::result::value || TypeIsSame<T, double>::result::value)
        {
            for (T* i = (T*)start; i != end; i++)
                *i = strtod(buffer, &buffer);
        }
        count = buffer - (char*)content.data();
    }

    static void read(std::istream& stream, void* start, const ::size_t& size)
    {
        T* end = (T*)start + size;
        for (T* i = (T*)start; i != end; i++)
            stream >> *i;
    }

    static void convert(void* source, float* destination, const ::size_t& size, const ::size_t& stride)
    {
        T* end = (T*)source + size * 3;
        for (T* i = (T*)source; i != end; i += 3)
        {
            destination[0] = (float)i[0];
            destination[1] = (float)i[1];
            destination[2] = (float)i[2];
            destination += stride;
        }
    }

    static void computeLogarithm(const ::size_t& histogramSize, T* histogramData)
    {
        T histogramMin(0.0f), histogramMax(0.0f);
        computeLogarithm(histogramSize, histogramData, histogramMin, histogramMax);
    }

    static void computeLogarithm(const ::size_t& histogramSize, T* histogramData, T& histogramMin, T& histogramMax)
    {
        histogramMin = FLT_MAX;
        histogramMax = 0.0f;
        T* histogramBegin = histogramData;
        T* histogramEnd = histogramBegin + histogramSize;
        const T logScale = 1.0f / log(2.0f);
        for (T *i = histogramBegin; i != histogramEnd; i++)
        {
            T t = *i = log(*i + 1.0f) * logScale;
            if (t < histogramMin) histogramMin = t;
            if (t > histogramMax) histogramMax = t;
        }

        T histogramScale = histogramMax - histogramMin < EPSILON ? 1.0f : 1.0f / (histogramMax - histogramMin);
        for (T *i = histogramBegin; i != histogramEnd; i++)
            *i = (*i - histogramMin) * histogramScale;
    }
    
    static void normalize(void* volumeData, const ::size_t& volumeSize, const QEndianness& endian, float& volumeMin, float& volumeMax)
    {
        T* begin = (T*)volumeData;
        T* end = begin + volumeSize;

        if (endian == ENDIAN_BIG)
        {
            for (T* i = begin; i != end; i++)
            {
                unsigned char* head = (unsigned char*)i;
                unsigned char* tail = head + sizeof(T) - 1;
                while (head < tail)
                {
                    unsigned char t = *head;
                    *(head++) = *tail;
                    *(tail--) = t;
                }
            }
        }

        volumeMin = FLT_MAX, volumeMax = -FLT_MAX;
        for (T* i = begin; i != end; i++)
        {
            float t = *i;
            if (t < volumeMin) volumeMin = t;
            if (t > volumeMax) volumeMax = t;
        }

        float scale = volumeMax - volumeMin < EPSILON ? 1.0f : 1.0f / (volumeMax - volumeMin);
        float *destination = (float *)volumeData + volumeSize;
        for (T *source = end; source != begin; )
            *(--destination) = (*(--source) - volumeMin) * scale;
    }
};

#endif  //QUTILITY_H