/**
 * COPYRIGHT NOTICE
 * Copyright (c) 2012, Institute of CG & CAD, Tsinghua University.
 * All Rights Reserved.
 * 
 * @file    QPipeline.h
 * @brief   QStage class definition.
 * 
 * This file defines the stage in visualization pipeline.
 * 
 * @version 1.0
 * @author  Jackie Pang
 * @e-mail: 15pengyi@gmail.com
 * @date    2012/02/07
 */

#ifndef QPIPELINE_H
#define QPIPELINE_H

#include <string>

#include <QThread>

enum StageState
{
    QCL_INITIALIZED     = 0,
    QCL_READ            = 1, 
    QCL_PREPROCESSED    = 2, 
    QCL_WRITTEN         = 3, 
    QCL_PAINTED         = 4
};

class QStage : public QThread
{
public:
    QStage(QObject* parent = NULL);
    ~QStage();

    virtual void init(const std::string& name);

protected:
    std::string name;
};

#endif  // QPIPELINE_H