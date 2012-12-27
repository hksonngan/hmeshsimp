/**
 * COPYRIGHT NOTICE
 * Copyright (c) 2012, Institute of CG & CAD, Tsinghua University.
 * All Rights Reserved.
 * 
 * @file    QStructure.h
 * @brief   QVector3 class, QVector4 class definition.
 * 
 * This file defines the data strctures commonly used in geometry processing.
 *     QVector3 class can be used as a 3D vector and represents mouse movement, view translation and so on.
 *     QVector4 class may be used as a 4D vector, which can represent view rotation.
 * 
 * @version 1.0
 * @author  Jackie Pang
 * @e-mail: 15pengyi@gmail.com
 * @date    2012/02/07
 */

#ifndef QSTRUCTURE_H
#define QSTRUCTURE_H

#define EPSILON         1e-5
#define PI              3.14259265359
#define BASE            10.0

#define FOVY            60.0
#define NEAR_CLIP       0.1
#define FAR_CLIP        100.0
#define MOUSE_SCALE     10.0
#define MAX_FRAMES      50

enum QMouseMode
{
    MOUSE_ROTATE    = 0,
    MOUSE_TRANSLATE = 1,
    MOUSE_DOLLY     = 2
};

enum QDataFormat
{
    DATA_UNKNOWN    = 0,
    DATA_UCHAR      = 1,
    DATA_USHORT     = 2,
    DATA_FLOAT      = 3,
    DATA_DOUBLE     = 4
};

enum QEndianness
{
    ENDIAN_BIG      = 0,
    ENDIAN_LITTLE   = 1
};

class QVector3
{
public:
    QVector3();
    QVector3(const QVector3 &v);
    QVector3(const float* data);
    QVector3(const float &x);
    QVector3(const float &x, const float &y, const float &z);
    ~QVector3();
    float x;
    float y;
    float z;
    
    friend QVector3 operator +(const QVector3 &v, const float &f);
    friend QVector3 operator -(const QVector3 &v, const float &f);
    friend QVector3 operator *(const QVector3 &v, const float &f);
    friend QVector3 operator /(const QVector3 &v, const float &f);

    friend QVector3 operator +(const float &f, const QVector3 &v);
    friend QVector3 operator -(const float &f, const QVector3 &v);
    friend QVector3 operator *(const float &f, const QVector3 &v);
    friend QVector3 operator /(const float &f, const QVector3 &v);
    
    friend QVector3 operator +(const QVector3 &v1, const QVector3 &v2);
    friend QVector3 operator -(const QVector3 &v1, const QVector3 &v2);
    friend QVector3 operator *(const QVector3 &v1, const QVector3 &v2);
    friend QVector3 operator /(const QVector3 &v1, const QVector3 &v2);
    
    static float dot(const QVector3 &v1, const QVector3 &v2);
    static QVector3 cross(const QVector3 &v1, const QVector3 &v2);
    static float length(const QVector3 &v);
    static QVector3 normalize(const QVector3 &v);
private:
    
};

class QVector4
{
public:
    QVector4();
    QVector4(const QVector4 &v);
    QVector4(float x, float y, float z, float w);
    ~QVector4();

    float x, y, z, w;
    
    friend QVector4 operator *(const QVector4 &t1, const QVector4 &t2);
    
    static void getAngleAxis(const QVector4 &q, QVector4 &v);
    static QVector4 fromAngleAxis(const float &angle, const QVector3 &axis);
    static QVector4 normalize(const QVector4 &q);
private:

};

#endif  // QSTRUCTURE_H