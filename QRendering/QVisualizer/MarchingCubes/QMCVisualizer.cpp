/**
 * COPYRIGHT NOTICE
 * Copyright (c) 2012, Institute of CG & CAD, Tsinghua University.
 * All Rights Reserved.
 * 
 * @file    QMCVisualizer.cpp
 * @brief   QMCVisualizer class declaration.
 * 
 * This file declares the initialization methods of components defined in QMCVisualizer.h.
 * 
 * @version 1.0
 * @author  Jackie Pang
 * @e-mail: 15pengyi@gmail.com
 * @date    2012/03/06
 */

#include "QMCVisualizer.h"

QMCVisualizer::QMCVisualizer(QWidget* parent) : QCommom(parent)
{
    initMenus();
    initWidgets();
}

QMCVisualizer::~QMCVisualizer()
{

}

void QMCVisualizer::initMenus()
{
    menuOpenFile_ = new QAction("Open Data File", parent_);
    menus_.file.push_front(menuOpenFile_);
}

void QMCVisualizer::initWidgets()
{
    panelControl_ = new QMCControlPanel(parent_);
    ((QTabWidget *)panel_)->addTab(panelControl_, "Control Panel");
    
    // E:/88Datasets/head/head.dat
    // E:/88Datasets/HurricanData/hurricane.dat
    // E:/88Datasets/2006/TS21z-S.dat
    // E:/88Datasets/DICOM/CARCINOMIX0.dat

	// F:/raw/raw_sets/head/head.dat
	// F:/raw/raw_sets/foot/foot.dat
	// F:/raw/raw_sets/ear/CTA_inOhr_1_128_char.dat
	// F:/raw/raw_sets/bluntfin/blunt_256x128x64.dat
	// F:/raw/raw_sets/cayley_cubic/d9.dat
	// F:/raw/raw_sets/ell64/ell64.dat
	// F:/raw/raw_sets/sphere/HohlKugel_64x64x64.dat
	// F:/raw/raw_sets/CT_128x128x53_char/CT_128x128x53_char.dat
	// F:/raw/raw_sets/Bonsai/Bonsai.dat
	// F:/raw/raw_sets/CT_256x256x106/CT_256x256x106.dat
	// F:/raw/raw_sets/Engine/Engine.dat
	// F:/raw/raw_sets/Head_256x256x256/Head_256x256x256.dat
	// F:/raw/raw_sets/MRI/MRI.dat
	// F:/raw/raw_sets/Teddybear/Teddybear.dat
	// F:/raw/raw_sets/lobster/lobster.dat
	// F:/raw/raw_sets/fuel/fuel.dat
	// F:/raw/raw_sets/statueLeg/statueLeg.dat
	// F:/raw/raw_sets/BostonTeapot/BostonTeapot.dat
	// F:/raw/raw_sets/skull/skull.dat
	// F:/raw/raw_sets/silicium/silicium.dat
	// F:/raw/raw_sets/nucleon/nucleon.dat
	// F:/raw/raw_sets/marschnerlobb/marschnerlobb.dat
	// F:/raw/raw_sets/aneurism/aneurism.dat

	//F:/raw/raw-volvis/vertebra16.dat
	//F:/raw/raw-volvis/vertebra8.dat

    render_ = new QMCWidget(parent_);
    ((QMCWidget *)render_)->initConnections(panelControl_);
	((QMCWidget *)render_)->initData("F:/raw/raw_sets/head/head.dat");
}