/**
 * COPYRIGHT NOTICE
 * Copyright (c) 2012, Institute of CG & CAD, Tsinghua University.
 * All Rights Reserved.
 * 
 * @file    QVSSetting.h
 * @brief   QVSSetting class definition.
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
 * @date    2012/03/19
 */

#ifndef QVSSETTING_H
#define QVSSETTING_H

#include "../infrastructures/QSetting.h"

#define NUMBER_VIEW_POINTS      200

#define CACHE_CL_BUFFER_SIZE    100 * 1024 * 1024
#define CACHE_CL_DEBUG_SIZE       4 * 1024 * 1024

class QVSSetting : public QSetting
{
public:
    QVSSetting();
    ~QVSSetting();

    GLboolean enableComputingEntropy;
    GLuint gaussian1D, gaussian2D;
};

#endif  // QVSSETTING_H