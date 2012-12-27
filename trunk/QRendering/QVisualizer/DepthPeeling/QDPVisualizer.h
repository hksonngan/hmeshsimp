/**
 * COPYRIGHT NOTICE
 * Copyright (c) 2012, Institute of CG & CAD, Tsinghua University.
 * All Rights Reserved.
 * 
 * @file    QDPVisualizer.h
 * @brief   QDPVisualizer class definition.
 * 
 * This file defines a subclass corresponds to a particular visualization algorithm named Time-varying Volumetric Data Visualization Framework.
 *     QDPVisualizer class also has four components(see the definition of QCommon),
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

#ifndef QDPVISUALIZER_H
#define QDPVISUALIZER_H

#include "ui_QDPControlPanel.h"

#include "../templates/QCommon.h"
#include "QDPWidget.h"
#include "QDPControlPanel.h"

class QDPVisualizer : public QCommom
{
    Q_OBJECT

public:
    QDPVisualizer(QWidget* parent = 0);
    ~QDPVisualizer();

protected:
    void initMenus();
    void initWidgets();

private:
    QDPControlPanel* panelControl_;
    QAction* menuOpenFile_;
};
#endif