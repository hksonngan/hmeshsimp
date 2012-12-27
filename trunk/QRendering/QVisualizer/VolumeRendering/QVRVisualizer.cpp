/**
 * COPYRIGHT NOTICE
 * Copyright (c) 2012, Institute of CG & CAD, Tsinghua University.
 * All Rights Reserved.
 * 
 * @file    QVRVisualizer.cpp
 * @brief   QVRVisualizer class declaration.
 * 
 * This file declares the initialization methods of components defined in QVRVisualizer.h.
 * 
 * @version 1.0
 * @author  Jackie Pang
 * @e-mail: 15pengyi@gmail.com
 * @date    2012/02/07
 */

#include "QVRVisualizer.h"

QVRVisualizer::QVRVisualizer(QWidget* parent) : QCommom(parent)
{
    initMenus();
    initWidgets();
}

QVRVisualizer::~QVRVisualizer()
{

}

void QVRVisualizer::initMenus()
{
    menuOpenFile_ = new QAction("Open Data File", parent_);
    menus_.file.push_front(menuOpenFile_);
}

void QVRVisualizer::initWidgets()
{
    panelTF1D_ = new QVRControlPanel(parent_);
    ((QTabWidget *)panel_)->addTab(panelTF1D_, "1D Transfer Function Design");
    
    // E:/88Datasets/head/head.dat
    // E:/88Datasets/HurricanData/hurricane.dat
    // E:/88Datasets/2006/TS21z.dat

    render_ = new QVRWidget(parent_);
    ((QVRWidget *)render_)->initConnections(panelTF1D_);
    ((QVRWidget *)render_)->initData("E:/88Datasets/2006/TS21z.dat");
}