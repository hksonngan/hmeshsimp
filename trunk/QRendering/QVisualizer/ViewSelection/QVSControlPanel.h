/**
 * COPYRIGHT NOTICE
 * Copyright (c) 2012, Institute of CG & CAD, Tsinghua University.
 * All Rights Reserved.
 * 
 * @file    QVSControlPanel.h
 * @brief   QVSControlPanel class definition.
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
 * @date    2012/03/19
 */

#ifndef QVSCONTROLPANEL_H
#define QVSCONTROLPANEL_H

#include <QtGui/QWidget>

#include "ui_QVSControlPanel.h"

class QVSControlPanel : public QWidget
{
    Q_OBJECT

public:
    QVSControlPanel(QWidget* parent = 0);
    ~QVSControlPanel();

    const Ui::QVSControlPanel* getUI();
    QColorDialog* colorDialog;
    int currentPushButton;

public slots:
    void slotUpdateBackgroundColor(const QColor &color);
    void slotSetPushButtonLightAmbient();
    void slotSetPushButtonLightDiffuse();
    void slotSetPushButtonLightSpecular();

private:
    Ui::QVSControlPanel ui;
};

#endif