/**
 * COPYRIGHT NOTICE
 * Copyright (c) 2012, Institute of CG & CAD, Tsinghua University.
 * All Rights Reserved.
 * 
 * @file    QDPSetting.h
 * @brief   QDPSetting class definition.
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

#ifndef QDPSETTING_H
#define QDPSETTING_H

#include "../infrastructures/QSetting.h"

#define CACHE_CL_BUFFER_SIZE    200 * 1024 * 1024
#define CACHE_CL_DEBUG_SIZE     1024

class QDPSetting : public QSetting
{
public:
    QDPSetting();
    ~QDPSetting();
};

#endif  // QFEATURESETTING