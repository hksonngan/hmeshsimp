/**
 * COPYRIGHT NOTICE
 * Copyright (c) 2012, Institute of CG & CAD, Tsinghua University.
 * All Rights Reserved.
 * 
 * @file    QSetting.h
 * @brief   QSetting class definition.
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

#ifndef QSETTING_H
#define QSETTING_H

#include <gl/glew.h>

#include <vector>

#include "../utilities/QUtility.h"
#include "../infrastructures/QStructure.h"

#define NUMBER_TF_ENTRIES   256
#define NUMBER_HIS_ENTRIES  256

// maximal memory size
#define CACHE_VOLUME_SIZE       600 * 1024 * 1024

#define STEPSIZE_DELTA  1.0 / 1024
#define STEPSIZE_MIN    1.0 / 1024
#define STEPSIZE_MAX    1.0 / 32

class QSetting
{
public:
    QSetting();
    ~QSetting();
    
    GLfloat volumeOffset, volumeScale, volumeStepSize;
    GLboolean enablePrintingFPS, enableSobelOperator, enableShading;
    
    QVector4 viewRotation;
    QVector3 viewTranslation;
    GLuint width, height;
    GLuint currentStep;

    QVector4 lightDirection;
    QVector4 lightSpecular;
    QVector4 lightDiffuse;
    QVector4 lightAmbient;
    GLfloat materialShininess;
    GLfloat diffuseCoeff;
    GLfloat ambientCoeff;
    GLfloat specularCoeff; 

    std::vector<GLsizei> transferFunctionSize;
    std::vector<GLfloat> transferFunctionData;
};

#endif  // QSETTING_H