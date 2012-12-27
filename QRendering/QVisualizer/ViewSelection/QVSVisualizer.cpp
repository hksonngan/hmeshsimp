/**
 * COPYRIGHT NOTICE
 * Copyright (c) 2012, Institute of CG & CAD, Tsinghua University.
 * All Rights Reserved.
 * 
 * @file    QVSVisualizer.cpp
 * @brief   QVSVisualizer class declaration.
 * 
 * This file declares the initialization methods of components defined in QVSVisualizer.h.
 * 
 * @version 1.0
 * @author  Jackie Pang
 * @e-mail: 15pengyi@gmail.com
 * @date    2012/03/19
 */

#include "QVSVisualizer.h"

QVSVisualizer::QVSVisualizer(QWidget* parent) : QCommom(parent)
{
    initMenus();
    initWidgets();
}

QVSVisualizer::~QVSVisualizer()
{

}

void QVSVisualizer::initMenus()
{
    menuOpenFile_ = new QAction("Open Data File", parent_);
    menus_.file.push_front(menuOpenFile_);
}

void QVSVisualizer::initWidgets()
{
    panelTF1D_ = new QVSControlPanel(parent_);
    ((QTabWidget *)panel_)->addTab(panelTF1D_, "1D Transfer Function Design");
    
    // E:/88Datasets/head/head.dat
    // E:/88Datasets/HurricanData/hurricane.dat
    // E:/88Datasets/2006/TS21z-S.dat

    render_ = new QVSWidget(parent_);
    ((QVSWidget *)render_)->initConnections(panelTF1D_);
    ((QVSWidget *)render_)->initData("E:/88Datasets/head/head.dat");
}