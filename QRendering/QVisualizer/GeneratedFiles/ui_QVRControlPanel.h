/********************************************************************************
** Form generated from reading UI file 'QVRControlPanel.ui'
**
** Created: Thu Dec 13 14:28:40 2012
**      by: Qt User Interface Compiler version 4.7.2
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_QVRCONTROLPANEL_H
#define UI_QVRCONTROLPANEL_H

#include <QtCore/QVariant>
#include <QtGui/QAction>
#include <QtGui/QApplication>
#include <QtGui/QButtonGroup>
#include <QtGui/QGroupBox>
#include <QtGui/QHBoxLayout>
#include <QtGui/QHeaderView>
#include <QtGui/QLabel>
#include <QtGui/QSlider>
#include <QtGui/QVBoxLayout>
#include <QtGui/QWidget>
#include "../infrastructures/QTransferFunction1D.h"

QT_BEGIN_NAMESPACE

class Ui_QVRControlPanel
{
public:
    QVBoxLayout *verticalLayout;
    QWidget *widgetTop;
    QHBoxLayout *horizontalLayoutTop;
    QWidget *widgetParameter;
    QVBoxLayout *verticalLayoutParameter;
    QGroupBox *groupBoxRayCasting;
    QVBoxLayout *verticalLayoutRayCasting;
    QWidget *widgetStepSize;
    QHBoxLayout *horizontalLayoutStepSize;
    QLabel *labelStepSize;
    QSlider *horizontalSliderStepSize;
    QLabel *labelStepSizeValue;
    QGroupBox *groupBoxRendering;
    QVBoxLayout *verticalLayoutRendering;
    QWidget *widgetVolumeScale;
    QHBoxLayout *horizontalLayoutVolumeScale;
    QLabel *labelVolumeScale;
    QSlider *horizontalSliderVolumeScale;
    QLabel *labelVolumeScaleValue;
    QWidget *widgetVolumeOffset;
    QHBoxLayout *horizontalLayoutVolumeOffset;
    QLabel *labelVolumeOffset;
    QSlider *horizontalSliderVolumeOffset;
    QLabel *labelVolumeOffsetValue;
    QTransferFunction1D *widgetEditor;
    QWidget *widgetBottom;
    QVBoxLayout *verticalLayout_2;
    QGroupBox *groupBoxTimeVaryingData;
    QHBoxLayout *horizontalLayoutTimeVaryingData;
    QLabel *labelTimeStep;
    QSlider *horizontalSliderTimeStep;
    QLabel *labelTimeStepValue;

    void setupUi(QWidget *QVRControlPanel)
    {
        if (QVRControlPanel->objectName().isEmpty())
            QVRControlPanel->setObjectName(QString::fromUtf8("QVRControlPanel"));
        QVRControlPanel->resize(530, 240);
        QVRControlPanel->setMinimumSize(QSize(0, 0));
        QVRControlPanel->setMaximumSize(QSize(16777215, 16777215));
        verticalLayout = new QVBoxLayout(QVRControlPanel);
#ifndef Q_OS_MAC
        verticalLayout->setSpacing(6);
#endif
        verticalLayout->setContentsMargins(3, 3, 3, 3);
        verticalLayout->setObjectName(QString::fromUtf8("verticalLayout"));
        widgetTop = new QWidget(QVRControlPanel);
        widgetTop->setObjectName(QString::fromUtf8("widgetTop"));
        horizontalLayoutTop = new QHBoxLayout(widgetTop);
#ifndef Q_OS_MAC
        horizontalLayoutTop->setSpacing(6);
#endif
        horizontalLayoutTop->setContentsMargins(3, 3, 3, 3);
        horizontalLayoutTop->setObjectName(QString::fromUtf8("horizontalLayoutTop"));
        widgetParameter = new QWidget(widgetTop);
        widgetParameter->setObjectName(QString::fromUtf8("widgetParameter"));
        widgetParameter->setMinimumSize(QSize(256, 0));
        widgetParameter->setMaximumSize(QSize(256, 16777215));
        widgetParameter->setStyleSheet(QString::fromUtf8(""));
        verticalLayoutParameter = new QVBoxLayout(widgetParameter);
#ifndef Q_OS_MAC
        verticalLayoutParameter->setSpacing(6);
#endif
        verticalLayoutParameter->setContentsMargins(0, 0, 0, 0);
        verticalLayoutParameter->setObjectName(QString::fromUtf8("verticalLayoutParameter"));
        groupBoxRayCasting = new QGroupBox(widgetParameter);
        groupBoxRayCasting->setObjectName(QString::fromUtf8("groupBoxRayCasting"));
        groupBoxRayCasting->setMinimumSize(QSize(0, 60));
        groupBoxRayCasting->setMaximumSize(QSize(16777215, 60));
        verticalLayoutRayCasting = new QVBoxLayout(groupBoxRayCasting);
#ifndef Q_OS_MAC
        verticalLayoutRayCasting->setSpacing(6);
#endif
        verticalLayoutRayCasting->setContentsMargins(6, 6, 6, 6);
        verticalLayoutRayCasting->setObjectName(QString::fromUtf8("verticalLayoutRayCasting"));
        widgetStepSize = new QWidget(groupBoxRayCasting);
        widgetStepSize->setObjectName(QString::fromUtf8("widgetStepSize"));
        horizontalLayoutStepSize = new QHBoxLayout(widgetStepSize);
        horizontalLayoutStepSize->setSpacing(0);
        horizontalLayoutStepSize->setContentsMargins(0, 0, 0, 0);
        horizontalLayoutStepSize->setObjectName(QString::fromUtf8("horizontalLayoutStepSize"));
        labelStepSize = new QLabel(widgetStepSize);
        labelStepSize->setObjectName(QString::fromUtf8("labelStepSize"));
        labelStepSize->setMinimumSize(QSize(96, 0));
        labelStepSize->setMaximumSize(QSize(96, 16777215));
        labelStepSize->setAlignment(Qt::AlignCenter);

        horizontalLayoutStepSize->addWidget(labelStepSize);

        horizontalSliderStepSize = new QSlider(widgetStepSize);
        horizontalSliderStepSize->setObjectName(QString::fromUtf8("horizontalSliderStepSize"));
        horizontalSliderStepSize->setMinimum(64);
        horizontalSliderStepSize->setMaximum(1024);
        horizontalSliderStepSize->setValue(256);
        horizontalSliderStepSize->setOrientation(Qt::Horizontal);

        horizontalLayoutStepSize->addWidget(horizontalSliderStepSize);

        labelStepSizeValue = new QLabel(widgetStepSize);
        labelStepSizeValue->setObjectName(QString::fromUtf8("labelStepSizeValue"));
        labelStepSizeValue->setMinimumSize(QSize(32, 0));
        labelStepSizeValue->setMaximumSize(QSize(32, 16777215));
        labelStepSizeValue->setAlignment(Qt::AlignRight|Qt::AlignTrailing|Qt::AlignVCenter);

        horizontalLayoutStepSize->addWidget(labelStepSizeValue);


        verticalLayoutRayCasting->addWidget(widgetStepSize);


        verticalLayoutParameter->addWidget(groupBoxRayCasting);

        groupBoxRendering = new QGroupBox(widgetParameter);
        groupBoxRendering->setObjectName(QString::fromUtf8("groupBoxRendering"));
        groupBoxRendering->setMinimumSize(QSize(0, 90));
        groupBoxRendering->setMaximumSize(QSize(16777215, 90));
        verticalLayoutRendering = new QVBoxLayout(groupBoxRendering);
#ifndef Q_OS_MAC
        verticalLayoutRendering->setSpacing(6);
#endif
        verticalLayoutRendering->setContentsMargins(6, 6, 6, 6);
        verticalLayoutRendering->setObjectName(QString::fromUtf8("verticalLayoutRendering"));
        widgetVolumeScale = new QWidget(groupBoxRendering);
        widgetVolumeScale->setObjectName(QString::fromUtf8("widgetVolumeScale"));
        horizontalLayoutVolumeScale = new QHBoxLayout(widgetVolumeScale);
        horizontalLayoutVolumeScale->setSpacing(0);
        horizontalLayoutVolumeScale->setContentsMargins(0, 0, 0, 0);
        horizontalLayoutVolumeScale->setObjectName(QString::fromUtf8("horizontalLayoutVolumeScale"));
        labelVolumeScale = new QLabel(widgetVolumeScale);
        labelVolumeScale->setObjectName(QString::fromUtf8("labelVolumeScale"));
        labelVolumeScale->setMinimumSize(QSize(96, 0));
        labelVolumeScale->setMaximumSize(QSize(96, 16777215));
        labelVolumeScale->setAlignment(Qt::AlignCenter);

        horizontalLayoutVolumeScale->addWidget(labelVolumeScale);

        horizontalSliderVolumeScale = new QSlider(widgetVolumeScale);
        horizontalSliderVolumeScale->setObjectName(QString::fromUtf8("horizontalSliderVolumeScale"));
        horizontalSliderVolumeScale->setMinimum(25);
        horizontalSliderVolumeScale->setMaximum(400);
        horizontalSliderVolumeScale->setPageStep(5);
        horizontalSliderVolumeScale->setValue(100);
        horizontalSliderVolumeScale->setOrientation(Qt::Horizontal);

        horizontalLayoutVolumeScale->addWidget(horizontalSliderVolumeScale);

        labelVolumeScaleValue = new QLabel(widgetVolumeScale);
        labelVolumeScaleValue->setObjectName(QString::fromUtf8("labelVolumeScaleValue"));
        labelVolumeScaleValue->setMinimumSize(QSize(32, 0));
        labelVolumeScaleValue->setMaximumSize(QSize(32, 16777215));
        labelVolumeScaleValue->setAlignment(Qt::AlignRight|Qt::AlignTrailing|Qt::AlignVCenter);

        horizontalLayoutVolumeScale->addWidget(labelVolumeScaleValue);


        verticalLayoutRendering->addWidget(widgetVolumeScale);

        widgetVolumeOffset = new QWidget(groupBoxRendering);
        widgetVolumeOffset->setObjectName(QString::fromUtf8("widgetVolumeOffset"));
        horizontalLayoutVolumeOffset = new QHBoxLayout(widgetVolumeOffset);
        horizontalLayoutVolumeOffset->setSpacing(0);
        horizontalLayoutVolumeOffset->setContentsMargins(0, 0, 0, 0);
        horizontalLayoutVolumeOffset->setObjectName(QString::fromUtf8("horizontalLayoutVolumeOffset"));
        labelVolumeOffset = new QLabel(widgetVolumeOffset);
        labelVolumeOffset->setObjectName(QString::fromUtf8("labelVolumeOffset"));
        labelVolumeOffset->setMinimumSize(QSize(96, 0));
        labelVolumeOffset->setMaximumSize(QSize(96, 16777215));
        labelVolumeOffset->setAlignment(Qt::AlignCenter);

        horizontalLayoutVolumeOffset->addWidget(labelVolumeOffset);

        horizontalSliderVolumeOffset = new QSlider(widgetVolumeOffset);
        horizontalSliderVolumeOffset->setObjectName(QString::fromUtf8("horizontalSliderVolumeOffset"));
        horizontalSliderVolumeOffset->setMinimum(-100);
        horizontalSliderVolumeOffset->setMaximum(100);
        horizontalSliderVolumeOffset->setPageStep(5);
        horizontalSliderVolumeOffset->setValue(0);
        horizontalSliderVolumeOffset->setOrientation(Qt::Horizontal);

        horizontalLayoutVolumeOffset->addWidget(horizontalSliderVolumeOffset);

        labelVolumeOffsetValue = new QLabel(widgetVolumeOffset);
        labelVolumeOffsetValue->setObjectName(QString::fromUtf8("labelVolumeOffsetValue"));
        labelVolumeOffsetValue->setMinimumSize(QSize(32, 0));
        labelVolumeOffsetValue->setMaximumSize(QSize(32, 16777215));
        labelVolumeOffsetValue->setAlignment(Qt::AlignRight|Qt::AlignTrailing|Qt::AlignVCenter);

        horizontalLayoutVolumeOffset->addWidget(labelVolumeOffsetValue);


        verticalLayoutRendering->addWidget(widgetVolumeOffset);


        verticalLayoutParameter->addWidget(groupBoxRendering);


        horizontalLayoutTop->addWidget(widgetParameter);

        widgetEditor = new QTransferFunction1D(widgetTop);
        widgetEditor->setObjectName(QString::fromUtf8("widgetEditor"));
        widgetEditor->setMinimumSize(QSize(256, 0));

        horizontalLayoutTop->addWidget(widgetEditor);


        verticalLayout->addWidget(widgetTop);

        widgetBottom = new QWidget(QVRControlPanel);
        widgetBottom->setObjectName(QString::fromUtf8("widgetBottom"));
        widgetBottom->setMinimumSize(QSize(0, 0));
        widgetBottom->setMaximumSize(QSize(16777215, 16777215));
        verticalLayout_2 = new QVBoxLayout(widgetBottom);
#ifndef Q_OS_MAC
        verticalLayout_2->setSpacing(6);
#endif
        verticalLayout_2->setContentsMargins(3, 3, 3, 3);
        verticalLayout_2->setObjectName(QString::fromUtf8("verticalLayout_2"));
        groupBoxTimeVaryingData = new QGroupBox(widgetBottom);
        groupBoxTimeVaryingData->setObjectName(QString::fromUtf8("groupBoxTimeVaryingData"));
        groupBoxTimeVaryingData->setMinimumSize(QSize(0, 60));
        groupBoxTimeVaryingData->setMaximumSize(QSize(16777215, 60));
        horizontalLayoutTimeVaryingData = new QHBoxLayout(groupBoxTimeVaryingData);
        horizontalLayoutTimeVaryingData->setContentsMargins(6, 6, 6, 6);
        horizontalLayoutTimeVaryingData->setObjectName(QString::fromUtf8("horizontalLayoutTimeVaryingData"));
        labelTimeStep = new QLabel(groupBoxTimeVaryingData);
        labelTimeStep->setObjectName(QString::fromUtf8("labelTimeStep"));
        labelTimeStep->setMinimumSize(QSize(96, 0));
        labelTimeStep->setMaximumSize(QSize(96, 16777215));
        labelTimeStep->setAlignment(Qt::AlignCenter);

        horizontalLayoutTimeVaryingData->addWidget(labelTimeStep);

        horizontalSliderTimeStep = new QSlider(groupBoxTimeVaryingData);
        horizontalSliderTimeStep->setObjectName(QString::fromUtf8("horizontalSliderTimeStep"));
        horizontalSliderTimeStep->setMinimum(0);
        horizontalSliderTimeStep->setMaximum(255);
        horizontalSliderTimeStep->setPageStep(0);
        horizontalSliderTimeStep->setOrientation(Qt::Horizontal);

        horizontalLayoutTimeVaryingData->addWidget(horizontalSliderTimeStep);

        labelTimeStepValue = new QLabel(groupBoxTimeVaryingData);
        labelTimeStepValue->setObjectName(QString::fromUtf8("labelTimeStepValue"));
        labelTimeStepValue->setMinimumSize(QSize(32, 0));
        labelTimeStepValue->setMaximumSize(QSize(32, 16777215));
        labelTimeStepValue->setAlignment(Qt::AlignRight|Qt::AlignTrailing|Qt::AlignVCenter);

        horizontalLayoutTimeVaryingData->addWidget(labelTimeStepValue);


        verticalLayout_2->addWidget(groupBoxTimeVaryingData);


        verticalLayout->addWidget(widgetBottom);


        retranslateUi(QVRControlPanel);
        QObject::connect(horizontalSliderStepSize, SIGNAL(valueChanged(int)), labelStepSizeValue, SLOT(setNum(int)));
        QObject::connect(horizontalSliderVolumeOffset, SIGNAL(valueChanged(int)), labelVolumeOffsetValue, SLOT(setNum(int)));
        QObject::connect(horizontalSliderVolumeScale, SIGNAL(valueChanged(int)), labelVolumeScaleValue, SLOT(setNum(int)));
        QObject::connect(horizontalSliderTimeStep, SIGNAL(valueChanged(int)), labelTimeStepValue, SLOT(setNum(int)));

        QMetaObject::connectSlotsByName(QVRControlPanel);
    } // setupUi

    void retranslateUi(QWidget *QVRControlPanel)
    {
        QVRControlPanel->setWindowTitle(QApplication::translate("QVRControlPanel", "Transfer Function Editor", 0, QApplication::UnicodeUTF8));
        groupBoxRayCasting->setTitle(QApplication::translate("QVRControlPanel", "1. Ray-casting", 0, QApplication::UnicodeUTF8));
        labelStepSize->setText(QApplication::translate("QVRControlPanel", "Step Size", 0, QApplication::UnicodeUTF8));
        labelStepSizeValue->setText(QApplication::translate("QVRControlPanel", "256", 0, QApplication::UnicodeUTF8));
        groupBoxRendering->setTitle(QApplication::translate("QVRControlPanel", "2. Volumetric Data", 0, QApplication::UnicodeUTF8));
        labelVolumeScale->setText(QApplication::translate("QVRControlPanel", "Volume Scale", 0, QApplication::UnicodeUTF8));
        labelVolumeScaleValue->setText(QApplication::translate("QVRControlPanel", "100", 0, QApplication::UnicodeUTF8));
        labelVolumeOffset->setText(QApplication::translate("QVRControlPanel", "Volume Offset", 0, QApplication::UnicodeUTF8));
        labelVolumeOffsetValue->setText(QApplication::translate("QVRControlPanel", "0", 0, QApplication::UnicodeUTF8));
        groupBoxTimeVaryingData->setTitle(QApplication::translate("QVRControlPanel", "3. Time-Varying Data", 0, QApplication::UnicodeUTF8));
        labelTimeStep->setText(QApplication::translate("QVRControlPanel", "Time Step", 0, QApplication::UnicodeUTF8));
        labelTimeStepValue->setText(QApplication::translate("QVRControlPanel", "0", 0, QApplication::UnicodeUTF8));
    } // retranslateUi

};

namespace Ui {
    class QVRControlPanel: public Ui_QVRControlPanel {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_QVRCONTROLPANEL_H
