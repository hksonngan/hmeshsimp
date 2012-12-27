/**
 * COPYRIGHT NOTICE
 * Copyright (c) 2012, Institute of CG & CAD, Tsinghua University.
 * All Rights Reserved.
 * 
 * @file    QVSVisualizer.h
 * @brief   QVSVisualizer class definition.
 * 
 * This file defines a subclass corresponds to a particular visualization algorithm named Time-varying Volumetric Data Visualization Framework.
 *     QVSVisualizer class also has four components(see the definition of QCommon),
 *         the parent widget is the main UI,
 *         the panel corresponds to a 1D Transfer Function Editor,
 *         the render is a standard volume render,
 *         and the menu list includes open file, save file and so on.
 * 
 * @version 1.0
 * @author  Edgar Liao, Jackie Pang
 * @e-mail: 15pengyi@gmail.com
 * @date    2012/03/19
 */

#ifndef QVSVISUALIZER_H
#define QVSVISUALIZER_H

#include "ui_QVSControlPanel.h"

#include "../templates/QCommon.h"
#include "QVSWidget.h"
#include "QVSControlPanel.h"

class QVSVisualizer : public QCommom
{
    Q_OBJECT

public:
    QVSVisualizer(QWidget* parent = 0);
    ~QVSVisualizer();

protected:
    void initMenus();
    void initWidgets();

private:
    QVSControlPanel* panelTF1D_;
    QAction* menuOpenFile_;
};
#endif