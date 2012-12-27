/**
 * COPYRIGHT NOTICE
 * Copyright (c) 2012, Institute of CG & CAD, Tsinghua University.
 * All Rights Reserved.
 * 
 * @file    QVRControlPanel.h
 * @brief   QVRControlPanel class definition.
 * 
 * This file defines a panel widget for users to adjust the parameters of Time-varying Volumetric Data Visualization Framework.
 *     The parameters includes
 *         the parameters of the ray-casting algorithm,
 *         volumetric data,
 *         transfer functions
 *         and time steps.
 * 
 * @version 1.0
 * @author  Jackie Pang
 * @e-mail: 15pengyi@gmail.com
 * @date    2012/02/07
 */

#ifndef QVRCONTROLPANEL_H
#define QVRCONTROLPANEL_H

#include <QtGui/QWidget>

#include "ui_QVRControlPanel.h"

class QVRControlPanel : public QWidget
{
public:
    QVRControlPanel(QWidget* parent = 0);
    ~QVRControlPanel();

    const Ui::QVRControlPanel* getUI();

private:
    Ui::QVRControlPanel ui;
};

#endif