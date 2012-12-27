/**
 * COPYRIGHT NOTICE
 * Copyright (c) 2012, Institute of CG & CAD, Tsinghua University.
 * All Rights Reserved.
 * 
 * @file    QVRControlPanel.cpp
 * @brief   QVRControlPanel class declaration.
 * 
 * This file declares the methods of the widget defined in QVRControlPanel.h.
 * 
 * @version 1.0
 * @author  Jackie Pang
 * @e-mail: 15pengyi@gmail.com
 * @date    2012/02/07
 */

#include "QVRControlPanel.h"

QVRControlPanel::QVRControlPanel(QWidget* parent) : QWidget(parent)
{
    ui.setupUi(this);
}

QVRControlPanel::~QVRControlPanel()
{

}

const Ui::QVRControlPanel* QVRControlPanel::getUI()
{
    return &ui;
}