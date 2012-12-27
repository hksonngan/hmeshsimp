/**
 * COPYRIGHT NOTICE
 * Copyright (c) 2012, Institute of CG & CAD, Tsinghua University.
 * All Rights Reserved.
 * 
 * @file    QMCControlPanel.h
 * @brief   QMCControlPanel class definition.
 * 
 * This file defines a panel widget for users to adjust the parameters of Marching Cubes.
 *     The parameters includes
 *         the iso-value of iso-surface.
 * 
 * @version 1.0
 * @author  Jackie Pang
 * @e-mail: 15pengyi@gmail.com
 * @date    2012/02/07
 */

#ifndef QMCCONTROLPANEL_H
#define QMCCONTROLPANEL_H

#include <QtGui/QWidget>

#include "ui_QMCControlPanel.h"

class QMCControlPanel : public QWidget
{
public:
    QMCControlPanel(QWidget* parent = 0);
    ~QMCControlPanel();

    const Ui::QMCControlPanel* getUI();

private:
    Ui::QMCControlPanel ui;
};

#endif