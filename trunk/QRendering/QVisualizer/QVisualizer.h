/**
 * COPYRIGHT NOTICE
 * Copyright (c) 2012, Institute of CG & CAD, Tsinghua University.
 * All Rights Reserved.
 * 
 * @file    QVisualizer.h
 * @brief   QVisualizer class definition.
 * 
 * This file defines two basic components and their initialization methods.
 * These componets includes a main UI and several visualization algorithms such as Volume Rendering, Marching Cubes and so on.
 * 
 * @version 1.0
 * @author  Edgar Liao, Jackie Pang
 * @e-mail: 15pengyi@gmail.com
 * @date    2012/02/07
 */

#ifndef QVISUALIZER_H
#define QVISUALIZER_H

#include <QtGui/QMainWindow>
#include <QtGui/QDockWidget>

#include "ui_QVisualizer.h"

class QCommom;

class QVisualizer : public QMainWindow
{
    Q_OBJECT

public:
    QVisualizer(QWidget *parent = 0, Qt::WFlags flags = 0);
    ~QVisualizer();

private:
    Ui::QVisualizerClass ui;
    QCommom* visualizer;
    
    void initMenus();
    void initPanels();
    void initConnections();

private slots:
    void slotInitVRVisualizer();
    void slotInitMCVisualizer();
    void slotInitMTVisualizer();
    void slotInitDPVisualizer();
    void slotInitVSVisualizer();
};

#endif // QVISUALIZER_H
