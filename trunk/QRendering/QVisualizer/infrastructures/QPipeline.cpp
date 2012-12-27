/**
 * COPYRIGHT NOTICE
 * Copyright (c) 2012, Institute of CG & CAD, Tsinghua University.
 * All Rights Reserved.
 * 
 * @file    QPipeline.cpp
 * @brief   QStage class declaration.
 * 
 * This file declares the commonly used methods defined in QPipeline.h.
 * 
 * @version 1.0
 * @author  Jackie Pang
 * @e-mail: 15pengyi@gmail.com
 * @date    2012/02/07
 */

#include "QPipeline.h"

QStage::QStage(QObject* parent) : QThread(parent),
    name()
{}

QStage::~QStage()
{}

void QStage::init(const std::string& name)
{}