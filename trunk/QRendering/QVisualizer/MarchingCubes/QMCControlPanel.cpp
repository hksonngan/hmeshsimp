/**
 * COPYRIGHT NOTICE
 * Copyright (c) 2012, Institute of CG & CAD, Tsinghua University.
 * All Rights Reserved.
 * 
 * @file    QMCControlPanel.cpp
 * @brief   QMCControlPanel class declaration.
 * 
 * This file declares the methods of the widget defined in QMCControlPanel.h.
 * 
 * @version 1.0
 * @author  Jackie Pang
 * @e-mail: 15pengyi@gmail.com
 * @date    2012/02/07
 */

#include "QMCControlPanel.h"

QMCControlPanel::QMCControlPanel(QWidget* parent) : QWidget(parent)
{
    ui.setupUi(this);
}

QMCControlPanel::~QMCControlPanel()
{

}

const Ui::QMCControlPanel* QMCControlPanel::getUI()
{
    return &ui;
}