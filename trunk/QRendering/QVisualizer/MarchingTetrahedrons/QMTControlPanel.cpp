/**
 * COPYRIGHT NOTICE
 * Copyright (c) 2012, Institute of CG & CAD, Tsinghua University.
 * All Rights Reserved.
 * 
 * @file    QMTControlPanel.cpp
 * @brief   QMTControlPanel class declaration.
 * 
 * This file declares the methods of the widget defined in QMTControlPanel.h.
 * 
 * @version 1.0
 * @author  Jackie Pang
 * @e-mail: 15pengyi@gmail.com
 * @date    2012/04/14
 */

#include "QMTControlPanel.h"

QMTControlPanel::QMTControlPanel(QWidget* parent) : QWidget(parent)
{
    ui.setupUi(this);
}

QMTControlPanel::~QMTControlPanel()
{

}

const Ui::QMTControlPanel* QMTControlPanel::getUI()
{
    return &ui;
}