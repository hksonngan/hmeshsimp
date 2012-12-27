/**
 * COPYRIGHT NOTICE
 * Copyright (c) 2012, Institute of CG & CAD, Tsinghua University.
 * All Rights Reserved.
 * 
 * @file    QMCVisualizer.h
 * @brief   QMCVisualizer class definition.
 * 
 * This file defines a subclass corresponds to a particular visualization algorithm named Time-varying Volumetric Data Visualization Framework.
 *     QMCVisualizer class also has four components(see the definition of QCommon),
 *         the parent widget is the main UI,
 *         the panel corresponds to a 1D Transfer Function Editor,
 *         the render is a standard volume render,
 *         and the menu list includes open file, save file and so on.
 * 
 * @version 1.0
 * @author  Edgar Liao, Jackie Pang
 * @e-mail: 15pengyi@gmail.com
 * @date    2012/03/06
 */

#ifndef QMCVISUALIZER_H
#define QMCVISUALIZER_H

#include "ui_QVRControlPanel.h"

#include "../templates/QCommon.h"
#include "QMCWidget.h"
#include "QMCControlPanel.h"

class QMCVisualizer : public QCommom
{
    Q_OBJECT

public:
    QMCVisualizer(QWidget* parent = 0);
    ~QMCVisualizer();

protected:
    void initMenus();
    void initWidgets();

private:
    QMCControlPanel* panelControl_;
    QAction* menuOpenFile_;
};
#endif