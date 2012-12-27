/**
 * COPYRIGHT NOTICE
 * Copyright (c) 2012, Institute of CG & CAD, Tsinghua University.
 * All Rights Reserved.
 * 
 * @file    QStructure.cpp
 * @brief   QVector3 class, QVector4 class declaration.
 * 
 * This file declares the commonly used methods of vectors defined in QStructure.h.
 * 
 * @version 1.0
 * @author  Jackie Pang
 * @e-mail: 15pengyi@gmail.com
 * @date    2012/02/07
 */

#include <cmath>

#include "QStructure.h"

QVector3::QVector3() :
    x(0.0f), y(0.0f), z(0.0f)
{}

QVector3::QVector3(const QVector3 &v) :
    x(v.x), y(v.y), z(v.z)
{}

QVector3::QVector3(const float *data) :
    x(data[0]), y(data[1]), z(data[2])
{}

QVector3::QVector3(const float &x) :
    x(x), y(x), z(x)
{}

QVector3::QVector3(const float &x, const float &y, const float &z) :
    x(x), y(y), z(z)
{}

QVector3::~QVector3()
{}

QVector3 operator +(const QVector3 &v, const float &f)
{
    return QVector3(v.x + f, v.y + f, v.z + f);
}

QVector3 operator -(const QVector3 &v, const float &f)
{
    return QVector3(v.x - f, v.y - f, v.z - f);
}

QVector3 operator *(const QVector3 & v, const float & f)
{
    return QVector3(v.x * f, v.y * f, v.z * f);
}

QVector3 operator /(const QVector3 & v, const float & f)
{
    float s = 1.0f / f;
    return QVector3(v.x * s, v.y * s, v.z * s);
}

QVector3 operator +(const float &f, const QVector3 &v)
{
    return QVector3(v.x + f, v.y + f, v.z + f);
}

QVector3 operator -(const float &f, const QVector3 &v)
{
    return QVector3(v.x - f, v.y - f, v.z - f);
}

QVector3 operator *(const float &f, const QVector3 &v)
{
    return QVector3(v.x * f, v.y * f, v.z * f);
}

QVector3 operator /(const float &f, const QVector3 &v)
{
    return QVector3(f / v.x, f / v.y, f / v.z);
}

QVector3 operator +(const QVector3 &v1, const QVector3 &v2)
{
    return QVector3(v1.x + v2.x, v1.y + v2.y, v1.z + v2.z);
}

QVector3 operator -(const QVector3 &v1, const QVector3 &v2)
{
    return QVector3(v1.x - v2.x, v1.y - v2.y, v1.z - v2.z);
}

QVector3 operator *(const QVector3 &v1, const QVector3 &v2)
{
    return QVector3(v1.x * v2.x, v1.y * v2.y, v1.z * v2.z);
}

QVector3 operator /(const QVector3 &v1, const QVector3 &v2)
{
    return QVector3(v1.x / v2.x, v1.y / v2.y, v1.z / v2.z);
}

float QVector3::dot(const QVector3 &v1, const QVector3 &v2)
{
    return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
}

QVector3 QVector3::cross(const QVector3 &v1, const QVector3 &v2)
{
    return QVector3(v1.y * v2.z - v1.z * v2.y, v1.z * v2.x - v1.x * v2.z, v1.x * v2.y - v1.y * v2.x);
}

float QVector3::length(const QVector3 &v)
{
    return std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
}

QVector3 QVector3::normalize(const QVector3 &v)
{
    float size = QVector3::length(v);
    if (size < EPSILON)
    {
        return QVector3(0.0f, 0.0f, 0.0f);
    }
    else
    {
        float scale = 1.0f / size;
        return QVector3(v.x * scale, v.y * scale, v.z * scale);
    }
}

QVector4::QVector4() :
    x(0.0f), y(0.0f), z(0.0f), w(0.0f)
{}

QVector4::QVector4(const QVector4 &v) :
    x(v.x), y(v.y), z(v.z), w(v.w)
{}

QVector4::QVector4(float x, float y, float z, float w) :
    x(x), y(y), z(z), w(w)
{}

QVector4::~QVector4()
{
    
}

QVector4 operator *(const QVector4 &t1, const QVector4 &t2)
{
    return QVector4(
            t1.w * t2.x + t2.w * t1.x + t1.y * t2.z - t1.z * t2.y,
            t1.w * t2.y + t2.w * t1.y + t1.z * t2.x - t1.x * t2.z,
            t1.w * t2.z + t2.w * t1.z + t1.x * t2.y - t1.y * t2.x,
            t1.w * t2.w - (t1.x * t2.x + t1.y * t2.y + t1.z * t2.z)
        );
}

void QVector4::getAngleAxis(const QVector4 &q, QVector4 &v)
{
    float size(std::sqrt(q.x * q.x + q.y * q.y + q.z * q.z));
    if (size < EPSILON)
    {
        v.x = 1.0f;
        v.y = v.z = v.w = 0.0f;
    }
    else
    {
        v.w = 2.0f * (float)::acos(q.w);
        v.x = q.x;
        v.y = q.y;
        v.z = q.z;
    }
}

QVector4 QVector4::fromAngleAxis(const float& angle, const QVector3 &axis)
{
    float size = QVector3::length(axis);
    if(size < EPSILON)
    {
        return QVector4(1.0f, 0.0f, 0.0f, 0.0f);
    }
    else
    {
        float scale = (float)::sin(angle * 0.5f) / size;
        return QVector4(axis.x * scale, axis.y * scale, axis.z * scale, (float)::cos(angle * 0.5f));
    }
}

QVector4 QVector4::normalize(const QVector4 &q)
{
    float size(std::sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w));
    if (size < EPSILON)
    {
        return QVector4(1.0f, 0.0f, 0.0f, 0.0f);
    }
    else
    {
        float scale = 1.0f / size;
        return QVector4(q.x * scale, q.y * scale, q.z * scale, q.w * scale);
    }
}