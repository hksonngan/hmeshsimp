/**
 * COPYRIGHT NOTICE
 * Copyright (c) 2012, Institute of CG & CAD, Tsinghua University.
 * All Rights Reserved.
 * 
 * @file    QMTVisualizer.h
 * @brief   QMTVisualizer class definition.
 * 
 * This file defines a subclass corresponds to a particular visualization algorithm named Time-varying Volumetric Data Visualization Framework.
 *     QMTVisualizer class also has four components(see the definition of QCommon),
 *         the parent widget is the main UI,
 *         the panel corresponds to a 1D Transfer Function Editor,
 *         the render is a standard volume render,
 *         and the menu list includes open file, save file and so on.
 * 
 * @version 1.0
 * @author  Edgar Liao, Jackie Pang
 * @e-mail: 15pengyi@gmail.com
 * @date    2012/04/14
 */

#ifndef QMTVISUALIZER_H
#define QMTVISUALIZER_H

#include "ui_QVRControlPanel.h"

#include "../templates/QCommon.h"
#include "QMTWidget.h"
#include "QMTControlPanel.h"

class QMTVisualizer : public QCommom
{
    Q_OBJECT

public:
    QMTVisualizer(QWidget* parent = 0);
    ~QMTVisualizer();

protected:
    void initMenus();
    void initWidgets();

private:
    QMTControlPanel* panelControl_;
    QAction* menuOpenFile_;
};
#endif