/********************************************************************************
** Form generated from reading UI file 'QDPControlPanel.ui'
**
** Created: Thu Dec 13 14:28:40 2012
**      by: Qt User Interface Compiler version 4.7.2
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_QDPCONTROLPANEL_H
#define UI_QDPCONTROLPANEL_H

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

class Ui_QDPControlPanel
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
    QGroupBox *groupBoxVolumetricData;
    QVBoxLayout *verticalLayoutVolumetricData;
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
    QGroupBox *groupBoxDepthPeeling;
    QVBoxLayout *verticalLayoutDepthPeeling;
    QWidget *widget;
    QHBoxLayout *horizontalLayoutColor;
    QLabel *labelColorTooltip;
    QSlider *horizontalSliderColor;
    QLabel *labelColorValue;
    QWidget *widgetAlpha;
    QHBoxLayout *horizontalLayoutAlpha;
    QLabel *labelAlphaTooltip;
    QSlider *horizontalSliderAlpha;
    QLabel *labelAlphaValue;
    QTransferFunction1D *widgetEditor;
    QWidget *widgetBottom;
    QVBoxLayout *verticalLayout_3;
    QGroupBox *groupBoxTimeVaryingData;
    QHBoxLayout *horizontalLayoutTimeVaryingData;
    QLabel *labelTimeStep;
    QSlider *horizontalSliderTimeStep;
    QLabel *labelTimeStepValue;

    void setupUi(QWidget *QDPControlPanel)
    {
        if (QDPControlPanel->objectName().isEmpty())
            QDPControlPanel->setObjectName(QString::fromUtf8("QDPControlPanel"));
        QDPControlPanel->resize(542, 348);
        QDPControlPanel->setMinimumSize(QSize(0, 0));
        QDPControlPanel->setMaximumSize(QSize(16777215, 16777215));
        verticalLayout = new QVBoxLayout(QDPControlPanel);
        verticalLayout->setObjectName(QString::fromUtf8("verticalLayout"));
        widgetTop = new QWidget(QDPControlPanel);
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

        groupBoxVolumetricData = new QGroupBox(widgetParameter);
        groupBoxVolumetricData->setObjectName(QString::fromUtf8("groupBoxVolumetricData"));
        groupBoxVolumetricData->setMinimumSize(QSize(0, 90));
        groupBoxVolumetricData->setMaximumSize(QSize(16777215, 90));
        verticalLayoutVolumetricData = new QVBoxLayout(groupBoxVolumetricData);
#ifndef Q_OS_MAC
        verticalLayoutVolumetricData->setSpacing(6);
#endif
        verticalLayoutVolumetricData->setContentsMargins(6, 6, 6, 6);
        verticalLayoutVolumetricData->setObjectName(QString::fromUtf8("verticalLayoutVolumetricData"));
        widgetVolumeScale = new QWidget(groupBoxVolumetricData);
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


        verticalLayoutVolumetricData->addWidget(widgetVolumeScale);

        widgetVolumeOffset = new QWidget(groupBoxVolumetricData);
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


        verticalLayoutVolumetricData->addWidget(widgetVolumeOffset);


        verticalLayoutParameter->addWidget(groupBoxVolumetricData);

        groupBoxDepthPeeling = new QGroupBox(widgetParameter);
        groupBoxDepthPeeling->setObjectName(QString::fromUtf8("groupBoxDepthPeeling"));
        groupBoxDepthPeeling->setMinimumSize(QSize(0, 90));
        groupBoxDepthPeeling->setMaximumSize(QSize(16777215, 90));
        verticalLayoutDepthPeeling = new QVBoxLayout(groupBoxDepthPeeling);
        verticalLayoutDepthPeeling->setContentsMargins(6, 6, 6, 6);
        verticalLayoutDepthPeeling->setObjectName(QString::fromUtf8("verticalLayoutDepthPeeling"));
        widget = new QWidget(groupBoxDepthPeeling);
        widget->setObjectName(QString::fromUtf8("widget"));
        widget->setMinimumSize(QSize(0, 32));
        widget->setMaximumSize(QSize(16777215, 32));
        horizontalLayoutColor = new QHBoxLayout(widget);
        horizontalLayoutColor->setSpacing(0);
        horizontalLayoutColor->setContentsMargins(0, 0, 0, 0);
        horizontalLayoutColor->setObjectName(QString::fromUtf8("horizontalLayoutColor"));
        labelColorTooltip = new QLabel(widget);
        labelColorTooltip->setObjectName(QString::fromUtf8("labelColorTooltip"));
        labelColorTooltip->setMinimumSize(QSize(96, 0));
        labelColorTooltip->setMaximumSize(QSize(96, 16777215));
        labelColorTooltip->setAlignment(Qt::AlignCenter);

        horizontalLayoutColor->addWidget(labelColorTooltip);

        horizontalSliderColor = new QSlider(widget);
        horizontalSliderColor->setObjectName(QString::fromUtf8("horizontalSliderColor"));
        horizontalSliderColor->setStyleSheet(QString::fromUtf8("background-color:\n"
"qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:0, \n"
"stop:0.0 hsv(000%, 80%, 80%),\n"
"stop:0.1 hsv(010%, 80%, 80%),\n"
"stop:0.2 hsv(020%, 80%, 80%),\n"
"stop:0.3 hsv(030%, 80%, 80%),\n"
"stop:0.4 hsv(040%, 80%, 80%),\n"
"stop:0.5 hsv(050%, 80%, 80%),\n"
"stop:0.6 hsv(060%, 80%, 80%),\n"
"stop:0.7 hsv(070%, 80%, 80%),\n"
"stop:0.8 hsv(080%, 80%, 80%),\n"
"stop:0.9 hsv(090%, 80%, 80%),\n"
"stop:1.0 hsv(100%, 80%, 80%)\n"
");"));
        horizontalSliderColor->setMinimum(0);
        horizontalSliderColor->setMaximum(200);
        horizontalSliderColor->setPageStep(5);
        horizontalSliderColor->setValue(100);
        horizontalSliderColor->setOrientation(Qt::Horizontal);

        horizontalLayoutColor->addWidget(horizontalSliderColor);

        labelColorValue = new QLabel(widget);
        labelColorValue->setObjectName(QString::fromUtf8("labelColorValue"));
        labelColorValue->setMinimumSize(QSize(32, 0));
        labelColorValue->setMaximumSize(QSize(32, 16777215));
        labelColorValue->setAlignment(Qt::AlignRight|Qt::AlignTrailing|Qt::AlignVCenter);

        horizontalLayoutColor->addWidget(labelColorValue);


        verticalLayoutDepthPeeling->addWidget(widget);

        widgetAlpha = new QWidget(groupBoxDepthPeeling);
        widgetAlpha->setObjectName(QString::fromUtf8("widgetAlpha"));
        widgetAlpha->setMinimumSize(QSize(0, 32));
        widgetAlpha->setMaximumSize(QSize(16777215, 32));
        horizontalLayoutAlpha = new QHBoxLayout(widgetAlpha);
        horizontalLayoutAlpha->setSpacing(0);
        horizontalLayoutAlpha->setContentsMargins(0, 0, 0, 0);
        horizontalLayoutAlpha->setObjectName(QString::fromUtf8("horizontalLayoutAlpha"));
        labelAlphaTooltip = new QLabel(widgetAlpha);
        labelAlphaTooltip->setObjectName(QString::fromUtf8("labelAlphaTooltip"));
        labelAlphaTooltip->setMinimumSize(QSize(96, 0));
        labelAlphaTooltip->setMaximumSize(QSize(96, 16777215));
        labelAlphaTooltip->setAlignment(Qt::AlignCenter);

        horizontalLayoutAlpha->addWidget(labelAlphaTooltip);

        horizontalSliderAlpha = new QSlider(widgetAlpha);
        horizontalSliderAlpha->setObjectName(QString::fromUtf8("horizontalSliderAlpha"));
        horizontalSliderAlpha->setMinimum(0);
        horizontalSliderAlpha->setMaximum(200);
        horizontalSliderAlpha->setPageStep(5);
        horizontalSliderAlpha->setValue(100);
        horizontalSliderAlpha->setOrientation(Qt::Horizontal);

        horizontalLayoutAlpha->addWidget(horizontalSliderAlpha);

        labelAlphaValue = new QLabel(widgetAlpha);
        labelAlphaValue->setObjectName(QString::fromUtf8("labelAlphaValue"));
        labelAlphaValue->setMinimumSize(QSize(32, 0));
        labelAlphaValue->setMaximumSize(QSize(32, 16777215));
        labelAlphaValue->setAlignment(Qt::AlignRight|Qt::AlignTrailing|Qt::AlignVCenter);

        horizontalLayoutAlpha->addWidget(labelAlphaValue);


        verticalLayoutDepthPeeling->addWidget(widgetAlpha);


        verticalLayoutParameter->addWidget(groupBoxDepthPeeling);


        horizontalLayoutTop->addWidget(widgetParameter);

        widgetEditor = new QTransferFunction1D(widgetTop);
        widgetEditor->setObjectName(QString::fromUtf8("widgetEditor"));
        widgetEditor->setMinimumSize(QSize(256, 0));

        horizontalLayoutTop->addWidget(widgetEditor);


        verticalLayout->addWidget(widgetTop);

        widgetBottom = new QWidget(QDPControlPanel);
        widgetBottom->setObjectName(QString::fromUtf8("widgetBottom"));
        widgetBottom->setMinimumSize(QSize(0, 0));
        widgetBottom->setMaximumSize(QSize(16777215, 16777215));
        verticalLayout_3 = new QVBoxLayout(widgetBottom);
#ifndef Q_OS_MAC
        verticalLayout_3->setSpacing(6);
#endif
        verticalLayout_3->setContentsMargins(3, 3, 3, 3);
        verticalLayout_3->setObjectName(QString::fromUtf8("verticalLayout_3"));
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


        verticalLayout_3->addWidget(groupBoxTimeVaryingData);


        verticalLayout->addWidget(widgetBottom);


        retranslateUi(QDPControlPanel);
        QObject::connect(horizontalSliderAlpha, SIGNAL(valueChanged(int)), labelAlphaValue, SLOT(setNum(int)));
        QObject::connect(horizontalSliderColor, SIGNAL(valueChanged(int)), labelColorValue, SLOT(setNum(int)));
        QObject::connect(horizontalSliderStepSize, SIGNAL(valueChanged(int)), labelTimeStepValue, SLOT(setNum(int)));
        QObject::connect(horizontalSliderVolumeScale, SIGNAL(valueChanged(int)), labelVolumeScaleValue, SLOT(setNum(int)));
        QObject::connect(horizontalSliderVolumeOffset, SIGNAL(valueChanged(int)), labelVolumeOffsetValue, SLOT(setNum(int)));
        QObject::connect(horizontalSliderTimeStep, SIGNAL(valueChanged(int)), labelTimeStepValue, SLOT(setNum(int)));

        QMetaObject::connectSlotsByName(QDPControlPanel);
    } // setupUi

    void retranslateUi(QWidget *QDPControlPanel)
    {
        QDPControlPanel->setWindowTitle(QApplication::translate("QDPControlPanel", "Control Panel", 0, QApplication::UnicodeUTF8));
        groupBoxRayCasting->setTitle(QApplication::translate("QDPControlPanel", "1. Ray-casting", 0, QApplication::UnicodeUTF8));
        labelStepSize->setText(QApplication::translate("QDPControlPanel", "Step Size", 0, QApplication::UnicodeUTF8));
        labelStepSizeValue->setText(QApplication::translate("QDPControlPanel", "256", 0, QApplication::UnicodeUTF8));
        groupBoxVolumetricData->setTitle(QApplication::translate("QDPControlPanel", "2. Volumetric Data", 0, QApplication::UnicodeUTF8));
        labelVolumeScale->setText(QApplication::translate("QDPControlPanel", "Volume Scale", 0, QApplication::UnicodeUTF8));
        labelVolumeScaleValue->setText(QApplication::translate("QDPControlPanel", "100", 0, QApplication::UnicodeUTF8));
        labelVolumeOffset->setText(QApplication::translate("QDPControlPanel", "Volume Offset", 0, QApplication::UnicodeUTF8));
        labelVolumeOffsetValue->setText(QApplication::translate("QDPControlPanel", "0", 0, QApplication::UnicodeUTF8));
        groupBoxDepthPeeling->setTitle(QApplication::translate("QDPControlPanel", "3. Depth Peeling", 0, QApplication::UnicodeUTF8));
        labelColorTooltip->setText(QApplication::translate("QDPControlPanel", "Color", 0, QApplication::UnicodeUTF8));
        labelColorValue->setText(QApplication::translate("QDPControlPanel", "100", 0, QApplication::UnicodeUTF8));
        labelAlphaTooltip->setText(QApplication::translate("QDPControlPanel", "Alpha", 0, QApplication::UnicodeUTF8));
        labelAlphaValue->setText(QApplication::translate("QDPControlPanel", "100", 0, QApplication::UnicodeUTF8));
        groupBoxTimeVaryingData->setTitle(QApplication::translate("QDPControlPanel", "4. Time-Varying Data", 0, QApplication::UnicodeUTF8));
        labelTimeStep->setText(QApplication::translate("QDPControlPanel", "Time Step", 0, QApplication::UnicodeUTF8));
        labelTimeStepValue->setText(QApplication::translate("QDPControlPanel", "0", 0, QApplication::UnicodeUTF8));
    } // retranslateUi

};

namespace Ui {
    class QDPControlPanel: public Ui_QDPControlPanel {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_QDPCONTROLPANEL_H
