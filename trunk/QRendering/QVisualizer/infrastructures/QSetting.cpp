/**
 * COPYRIGHT NOTICE
 * Copyright (c) 2012, Institute of CG & CAD, Tsinghua University.
 * All Rights Reserved.
 * 
 * @file    QSetting.cpp
 * @brief   QSetting class declaration.
 * 
 * This file declares the often used settings defined in QSetting.h.
 * 
 * @version 1.0
 * @author  Jackie Pang
 * @e-mail: 15pengyi@gmail.com
 * @date    2012/03/19
 */

#include <limits>

#include "QSetting.h"

QSetting::QSetting() :
    width(512), height(512), currentStep(0),
    viewRotation(0.0f, 0.0f, 0.0f, 1.0f), viewTranslation(0.0f, 0.0f, -8.0f),
    lightDirection(1.0f, 1.0f, 1.0f, 0.0f), lightSpecular(0.1f, 0.1f, 0.1f, 0.0f), lightDiffuse(0.5f, 0.5f, 0.5f, 0.0f), lightAmbient(1.0f, 1.0f, 1.0f, 1.0f), materialShininess(2.0f),
    diffuseCoeff(0.3f), ambientCoeff(0.8f), specularCoeff(0.1f),
    enableSobelOperator(GL_FALSE), enablePrintingFPS(GL_TRUE), enableShading(GL_FALSE),
    volumeStepSize(0.01f), volumeOffset(0.0f), volumeScale(1.0f),
    transferFunctionSize(1, NUMBER_TF_ENTRIES), transferFunctionData(NUMBER_TF_ENTRIES * 4, float(0.0f))
{}

QSetting::~QSetting()
{}