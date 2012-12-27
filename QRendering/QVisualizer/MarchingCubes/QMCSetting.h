/**
 * COPYRIGHT NOTICE
 * Copyright (c) 2012, Institute of CG & CAD, Tsinghua University.
 * All Rights Reserved.
 * 
 * @file    QMCSetting.h
 * @brief   QMCSetting class definition.
 * 
 * This file defines the most often used settings such as
 *     volume offset and volume scale(also known as window width and window level),
 *     geometry(translation and rotation),
 *     illumination parameters,
 *     and transfer functions.
 * 
 * @version 1.0
 * @author  Jackie Pang
 * @e-mail: 15pengyi@gmail.com
 * @date    2012/03/06
 */

#ifndef QMCSETTING_H
#define QMCSETTING_H

#include "../infrastructures/QSetting.h"

#define CACHE_CL_BUFFER_SIZE    200 * 1024 * 1024
#define CACHE_CL_DEBUG_SIZE     1024

#define ISOVALUE_DELTA  0.005f
#define ISOVALUE_MIN    0.0f
#define ISOVALUE_MAX    1.0f

class QMCSetting : public QSetting
{
public:
    QMCSetting();
    ~QMCSetting();
    
    GLfloat isoValue;
};

#endif  // QFEATURESETTING