/**
 * COPYRIGHT NOTICE
 * Copyright (c) 2012, Institute of CG & CAD, Tsinghua University.
 * All Rights Reserved.
 * 
 * @file    QDPControlPanel.cpp
 * @brief   QDPControlPanel class declaration.
 * 
 * This file declares the methods of the widget defined in QDPControlPanel.h.
 * 
 * @version 1.0
 * @author  Jackie Pang
 * @e-mail: 15pengyi@gmail.com
 * @date    2012/02/07
 */

#include "QDPControlPanel.h"

QDPControlPanel::QDPControlPanel(QWidget* parent) : QWidget(parent)
{
    ui.setupUi(this);
}

QDPControlPanel::~QDPControlPanel()
{

}

const Ui::QDPControlPanel* QDPControlPanel::getUI()
{
    return &ui;
}