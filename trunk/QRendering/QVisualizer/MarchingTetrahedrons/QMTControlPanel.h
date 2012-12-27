/**
 * COPYRIGHT NOTICE
 * Copyright (c) 2012, Institute of CG & CAD, Tsinghua University.
 * All Rights Reserved.
 * 
 * @file    QMTControlPanel.h
 * @brief   QMTControlPanel class definition.
 * 
 * This file defines a panel widget for users to adjust the parameters of Marching Tetrahedrons.
 *     The parameters includes
 *         the iso-value of iso-surface.
 * 
 * @version 1.0
 * @author  Jackie Pang
 * @e-mail: 15pengyi@gmail.com
 * @date    2012/04/14
 */

#ifndef QMTCONTROLPANEL_H
#define QMTCONTROLPANEL_H

#include <QtGui/QWidget>

#include "ui_QMTControlPanel.h"

class QMTControlPanel : public QWidget
{
public:
    QMTControlPanel(QWidget* parent = 0);
    ~QMTControlPanel();

    const Ui::QMTControlPanel* getUI();

private:
    Ui::QMTControlPanel ui;
};

#endif