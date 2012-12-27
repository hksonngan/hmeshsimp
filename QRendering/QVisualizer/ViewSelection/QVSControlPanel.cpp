/**
 * COPYRIGHT NOTICE
 * Copyright (c) 2012, Institute of CG & CAD, Tsinghua University.
 * All Rights Reserved.
 * 
 * @file    QVSControlPanel.cpp
 * @brief   QVSControlPanel class declaration.
 * 
 * This file declares the methods of the widget defined in QVSControlPanel.h.
 * 
 * @version 1.0
 * @author  Jackie Pang
 * @e-mail: 15pengyi@gmail.com
 * @date    2012/03/19
 */

#include <sstream>

#include <QColor>
#include <QColorDialog>

#include "QVSControlPanel.h"

QVSControlPanel::QVSControlPanel(QWidget* parent) : QWidget(parent),
    currentPushButton(-1)
{
    ui.setupUi(this);

    colorDialog = new QColorDialog(this);

    connect(ui.pushButtonLightAmbientValue, SIGNAL(clicked()), this, SLOT(slotSetPushButtonLightAmbient()));
    connect(ui.pushButtonLightAmbientValue, SIGNAL(clicked()), colorDialog, SLOT(open()));
    connect(ui.pushButtonLightDiffuseValue, SIGNAL(clicked()), this, SLOT(slotSetPushButtonLightDiffuse()));
    connect(ui.pushButtonLightDiffuseValue, SIGNAL(clicked()), colorDialog, SLOT(open()));
    connect(ui.pushButtonLightSpecularValue, SIGNAL(clicked()), this, SLOT(slotSetPushButtonLightSpecular()));
    connect(ui.pushButtonLightSpecularValue, SIGNAL(clicked()), colorDialog, SLOT(open()));

    connect(colorDialog, SIGNAL(colorSelected(const QColor&)), this, SLOT(slotUpdateBackgroundColor(const QColor&)));
}

QVSControlPanel::~QVSControlPanel()
{

}

const Ui::QVSControlPanel* QVSControlPanel::getUI()
{
    return &ui;
}

void QVSControlPanel::slotUpdateBackgroundColor(const QColor &color)
{
    std::stringstream styleSheet;
    styleSheet << "border: 1px solid gray;\nbackground-color: rgb(";
    styleSheet << color.red();
    styleSheet << ",";
    styleSheet << color.green();
    styleSheet << ",";
    styleSheet << color.blue();
    styleSheet << ")";

    QPushButton* button(NULL);
    switch (currentPushButton)
    {
    case 0:
        button = ui.pushButtonLightAmbientValue;
    	break;
    case 1:
        button = ui.pushButtonLightDiffuseValue;
        break;
    case 2:
        button = ui.pushButtonLightSpecularValue;
        break;
    }
    if (button) button->setStyleSheet(QString(styleSheet.str().c_str()));
}

void QVSControlPanel::slotSetPushButtonLightAmbient()
{
    currentPushButton = 0;
}

void QVSControlPanel::slotSetPushButtonLightDiffuse()
{
    currentPushButton = 1;
}

void QVSControlPanel::slotSetPushButtonLightSpecular()
{
    currentPushButton = 2;
}