/**
 * COPYRIGHT NOTICE
 * Copyright (c) 2012, Institute of CG & CAD, Tsinghua University.
 * All Rights Reserved.
 * 
 * @file    QVSSetting.cpp
 * @brief   QVSSetting class declaration.
 * 
 * This file declares the often used settings defined in QVSSetting.h.
 * 
 * @version 1.0
 * @author  Jackie Pang
 * @e-mail: 15pengyi@gmail.com
 * @date    2012/03/19
 */

#include "QVSSetting.h"

QVSSetting::QVSSetting() : QSetting(),
    enableComputingEntropy(GL_FALSE),
    gaussian1D(3), gaussian2D(3)
{}

QVSSetting::~QVSSetting()
{}