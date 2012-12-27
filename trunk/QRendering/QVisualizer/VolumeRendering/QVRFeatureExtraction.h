/**
 * COPYRIGHT NOTICE
 * Copyright (c) 2012, Institute of CG & CAD, Tsinghua University.
 * All Rights Reserved.
 * 
 * @file    *.h
 * @brief   * class definition.
 * 
 * This file defines *.
 * 
 * @version 1.0
 * @author  Jackie Pang
 * @e-mail: 15pengyi@gmail.com
 * @date    2012/02/07
 */

#ifndef QVRFEATUREEXTRACTION_H
#define QVRFEATUREEXTRACTION_H

#include <cl/cl.h>

#include "../infrastructures/QCLProgram.h"

class QVRFeatureExtracation
{
public:
    QVRFeatureExtracation();
    ~QVRFeatureExtracation();

    static unsigned char extractFeatures(const cl_context& clContext, std::list<QCLProgram>& clPrograms, cl_command_queue& clQueue, unsigned char* volumeData, unsigned int* volumeSize, int voxelSize);
};

#endif  // QVRFEATUREEXTRACTION_H