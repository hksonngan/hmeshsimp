/**
 * COPYRIGHT NOTICE
 * Copyright (c) 2012, Institute of CG & CAD, Tsinghua University.
 * All Rights Reserved.
 * 
 * @file    QVRSetting.h
 * @brief   QVRSetting class definition.
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
 * @date    2012/02/07
 */

#ifndef QVRSETTING_H
#define QVRSETTING_H

#include <gl/glew.h>

#include <vector>

#include "../infrastructures/QSetting.h"

#define CACHE_CL_VOLUME_SIZE    300 * 1024 * 1024
#define CACHE_CL_DEBUG_SIZE        1024

class QVRSetting : public QSetting
{
public:
    QVRSetting();
    ~QVRSetting();

    GLboolean enablePrintingBandwidth;
};

#endif  // QVRSETTING_H