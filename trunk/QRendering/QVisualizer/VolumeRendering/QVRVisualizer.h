/**
 * COPYRIGHT NOTICE
 * Copyright (c) 2012, Institute of CG & CAD, Tsinghua University.
 * All Rights Reserved.
 * 
 * @file    QVRVisualizer.h
 * @brief   QVRVisualizer class definition.
 * 
 * This file defines a subclass corresponds to a particular visualization algorithm named Time-varying Volumetric Data Visualization Framework.
 *     QVRVisualizer class also has four components(see the definition of QCommon),
 *         the parent widget is the main UI,
 *         the panel corresponds to a 1D Transfer Function Editor,
 *         the render is a standard volume render,
 *         and the menu list includes open file, save file and so on.
 * 
 * @version 1.0
 * @author  Edgar Liao, Jackie Pang
 * @e-mail: 15pengyi@gmail.com
 * @date    2012/02/07
 */

#ifndef QVRVISUALIZER_H
#define QVRVISUALIZER_H

#include "ui_QVRControlPanel.h"

#include "../templates/QCommon.h"
#include "QVRWidget.h"
#include "QVRControlPanel.h"

class QVRVisualizer : public QCommom
{
    Q_OBJECT

public:
    QVRVisualizer(QWidget* parent = 0);
    ~QVRVisualizer();

protected:
    void initMenus();
    void initWidgets();

private:
    QVRControlPanel* panelTF1D_;
    QAction* menuOpenFile_;
};
#endif