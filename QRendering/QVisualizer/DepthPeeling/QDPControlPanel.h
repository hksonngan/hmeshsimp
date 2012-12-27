/**
 * COPYRIGHT NOTICE
 * Copyright (c) 2012, Institute of CG & CAD, Tsinghua University.
 * All Rights Reserved.
 * 
 * @file    QDPControlPanel.h
 * @brief   QDPControlPanel class definition.
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

#ifndef QDPCONTROLPANEL_H
#define QDPCONTROLPANEL_H

#include <QtGui/QWidget>

#include "ui_QDPControlPanel.h"

class QDPControlPanel : public QWidget
{
public:
    QDPControlPanel(QWidget* parent = 0);
    ~QDPControlPanel();

    const Ui::QDPControlPanel* getUI();

private:
    Ui::QDPControlPanel ui;
};

#endif