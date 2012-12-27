/**
 * COPYRIGHT NOTICE
 * Copyright (c) 2012, Institute of CG & CAD, Tsinghua University.
 * All Rights Reserved.
 * 
 * @file    QDPVisualizer.cpp
 * @brief   QDPVisualizer class declaration.
 * 
 * This file declares the initialization methods of components defined in QDPVisualizer.h.
 * 
 * @version 1.0
 * @author  Jackie Pang
 * @e-mail: 15pengyi@gmail.com
 * @date    2012/03/06
 */

#include "QDPVisualizer.h"

QDPVisualizer::QDPVisualizer(QWidget* parent) : QCommom(parent)
{
    initMenus();
    initWidgets();
}

QDPVisualizer::~QDPVisualizer()
{

}

void QDPVisualizer::initMenus()
{
    menuOpenFile_ = new QAction("Open Data File", parent_);
    menus_.file.push_front(menuOpenFile_);
}

void QDPVisualizer::initWidgets()
{
    panelControl_ = new QDPControlPanel(parent_);
    ((QTabWidget *)panel_)->addTab(panelControl_, "Control Panel");
    
    // E:/88Datasets/head/head.dat
    // E:/88Datasets/HurricanData/hurricane.dat
    // E:/88Datasets/2006/TS21z-S.dat
    // E:/88Datasets/DICOM/CARCINOMIX0.dat

    render_ = new QDPWidget(parent_);
    ((QDPWidget *)render_)->initConnections(panelControl_);
    ((QDPWidget *)render_)->initData("E:/88Datasets/2006/TS21z-S.dat");
}