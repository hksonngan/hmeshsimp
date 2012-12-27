/**
 * COPYRIGHT NOTICE
 * Copyright (c) 2012, Institute of CG & CAD, Tsinghua University.
 * All Rights Reserved.
 * 
 * @file    QMTVisualizer.cpp
 * @brief   QMTVisualizer class declaration.
 * 
 * This file declares the initialization methods of components defined in QMTVisualizer.h.
 * 
 * @version 1.0
 * @author  Jackie Pang
 * @e-mail: 15pengyi@gmail.com
 * @date    2012/04/14
 */

#include "QMTVisualizer.h"

QMTVisualizer::QMTVisualizer(QWidget* parent) : QCommom(parent)
{
    initMenus();
    initWidgets();
}

QMTVisualizer::~QMTVisualizer()
{

}

void QMTVisualizer::initMenus()
{
    menuOpenFile_ = new QAction("Open Data File", parent_);
    menus_.file.push_front(menuOpenFile_);
}

void QMTVisualizer::initWidgets()
{
    panelControl_ = new QMTControlPanel(parent_);
    ((QTabWidget *)panel_)->addTab(panelControl_, "Control Panel");
    
    // E:/88Datasets/head/head.dat
    // E:/88Datasets/HurricanData/hurricane.dat
    // E:/88Datasets/2006/TS21z-S.dat
    // E:/88Datasets/DICOM/CARCINOMIX0.dat
    // E:/88Datasets/Tokyo/Tanaka/mergef_wg01--0100_vof_func.dat
    // E:/88Datasets/Tokyo/Tanaka/mergef_wg01--0100_Pressure.dat

    render_ = new QMTWidget(parent_);
    ((QMTWidget *)render_)->initConnections(panelControl_);
    ((QMTWidget *)render_)->initData("E:/88Datasets/Tokyo/Tanaka/mergef_wg01--0100_Pressure.dat");
}