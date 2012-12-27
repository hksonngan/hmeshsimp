/**
 * COPYRIGHT NOTICE
 * Copyright (c) 2012, Institute of CG & CAD, Tsinghua University.
 * All Rights Reserved.
 * 
 * @file    QVRPipeline.h
 * @brief   QVRReader class, QVRPreprocessor class and QVRWriter class definition.
 * 
 * This file declares all the stages in visualization pipeline(see QVRWidget.h), each stage corresponds to a particular class derived from QThread.
 *     QVRReader class is in charge of loading volume data from disk.
 *     QVRPreprocessor class does all the data preprocessing jobs.
 *     QVRWriter class takes charge of data transmission from CPU to GPU.
 * 
 * @version 1.0
 * @author  Jackie Pang
 * @e-mail: 15pengyi@gmail.com
 * @date    2012/02/07
 */

#ifndef QVRPIPELINE_H
#define QVRPIPELINE_H

#include <string>

#include "../infrastructures/QPipeline.h"

class QVRWidget;

class QVRReader : public QStage
{
public:
    QVRReader(QVRWidget* parent = NULL);
    ~QVRReader();

    void init(const std::string& name);
    void run();

private:
    QVRWidget* parent;
    std::string name;
};

class QVRPreprocessor : public QStage
{
public:
    QVRPreprocessor(QVRWidget* parent = NULL);
    ~QVRPreprocessor();
    
    void run();

private:
    QVRWidget* parent;
};

class QVRWriter : public QStage
{
public:
    QVRWriter(QVRWidget* parent = NULL);
    ~QVRWriter();

    void run();

private:
    QVRWidget* parent;
};

#endif  // QVRPIPELINE_H