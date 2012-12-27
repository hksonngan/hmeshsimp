/**
 * COPYRIGHT NOTICE
 * Copyright (c) 2012, Institute of CG & CAD, Tsinghua University.
 * All Rights Reserved.
 * 
 * @file    QVisualizer.cpp
 * @brief   QVisualizer class declaration.
 * 
 * This file declares the initialization methods of components defined in QVisualizer.h.
 * 
 * @version 1.0
 * @author  Jackie Pang
 * @e-mail: 15pengyi@gmail.com
 * @date    2012/02/07
 */

#include <QtGui/QVBoxLayout>

#include "QVisualizer.h"
#include "VolumeRendering/QVRVisualizer.h"
#include "MarchingCubes/QMCVisualizer.h"
#include "MarchingTetrahedrons/QMTVisualizer.h"
#include "DepthPeeling/QDPVisualizer.h"
#include "ViewSelection//QVSVisualizer.h"

QVisualizer::QVisualizer(QWidget *parent, Qt::WFlags flags) : QMainWindow(parent, flags)
{
    ui.setupUi(this);
    
    initPanels();
    initConnections();
}

QVisualizer::~QVisualizer()
{

}

void QVisualizer::slotInitVRVisualizer()
{
    visualizer = new QVRVisualizer(this);

    initMenus();
}

void QVisualizer::slotInitMCVisualizer()
{
    visualizer = new QMCVisualizer(this);

    initMenus();
}

void QVisualizer::slotInitMTVisualizer()
{
    visualizer = new QMTVisualizer(this);

    initMenus();
}

void QVisualizer::slotInitDPVisualizer()
{
    visualizer = new QDPVisualizer(this);

    initMenus();
}

void QVisualizer::slotInitVSVisualizer()
{
    visualizer = new QVSVisualizer(this);

    initMenus();
}

void QVisualizer::initMenus()
{
    Menu* menus = visualizer->getMenus();
    for (std::list<QAction *>::iterator i = menus->file.begin(); i != menus->file.end(); i++)
        ui.menuFile->addAction(*i);
    for (std::list<QAction *>::iterator i = menus->edit.begin(); i != menus->edit.end(); i++)
        ui.menuEdit->addAction(*i);
    for (std::list<QAction *>::iterator i = menus->view.begin(); i != menus->view.end(); i++)
        ui.menuView->addAction(*i);
    for (std::list<QAction *>::iterator i = menus->settings.begin(); i != menus->settings.end(); i++)
        ui.menuSettings->addAction(*i);
}
 
void QVisualizer::initPanels()
{
    // for debugging
    // slotInitVRVisualizer();
    slotInitMCVisualizer();
    // slotInitMTVisualizer();
    // slotInitDPVisualizer();
    // slotInitVSVisualizer();
    
    QVBoxLayout* layout = new QVBoxLayout;
    layout->addWidget(visualizer->getRender());
    ui.centralwidget->setLayout(layout);
    ui.dockWidget->setWidget(visualizer->getPanel());
}

void QVisualizer::initConnections()
{
    connect(ui.actionMarching_Cubes, SIGNAL(triggered()), this, SLOT(slotInitVRVisualizer()));
    connect(ui.actionVolume_Rendering, SIGNAL(triggered()), this, SLOT(slotInitMCVisualizer()));
}