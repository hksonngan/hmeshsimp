/********************************************************************************
** Form generated from reading UI file 'QVSControlPanel.ui'
**
** Created: Thu Dec 13 14:28:40 2012
**      by: Qt User Interface Compiler version 4.7.2
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_QVSCONTROLPANEL_H
#define UI_QVSCONTROLPANEL_H

#include <QtCore/QVariant>
#include <QtGui/QAction>
#include <QtGui/QApplication>
#include <QtGui/QButtonGroup>
#include <QtGui/QDoubleSpinBox>
#include <QtGui/QGroupBox>
#include <QtGui/QHBoxLayout>
#include <QtGui/QHeaderView>
#include <QtGui/QLabel>
#include <QtGui/QPushButton>
#include <QtGui/QRadioButton>
#include <QtGui/QSlider>
#include <QtGui/QSpacerItem>
#include <QtGui/QVBoxLayout>
#include <QtGui/QWidget>
#include "../infrastructures/QHemisphere.h"
#include "../infrastructures/QTransferFunction1D.h"

QT_BEGIN_NAMESPACE

class Ui_QVSControlPanel
{
public:
    QVBoxLayout *verticalLayout;
    QWidget *widgetTop;
    QHBoxLayout *horizontalLayoutTop;
    QWidget *widgetLeft;
    QVBoxLayout *verticalLayoutLeft;
    QGroupBox *groupBoxRayCasting;
    QVBoxLayout *verticalLayoutRayCasting;
    QWidget *widgetStepSize;
    QHBoxLayout *horizontalLayoutStepSize;
    QLabel *labelStepSize;
    QSlider *horizontalSliderStepSize;
    QLabel *labelStepSizeValue;
    QSpacerItem *verticalSpacerLeft2;
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
    QSpacerItem *verticalSpacerLeft1;
    QGroupBox *groupBoxIlluminationModel;
    QVBoxLayout *verticalLayoutIlluminationModel;
    QWidget *widgetLightPosition;
    QHBoxLayout *horizontalLayoutLightPosition;
    QLabel *labelLightPosition;
    QSpacerItem *horizontalSpacerLightPosition;
    QLabel *labelLightPositionX;
    QDoubleSpinBox *doubleSpinBoxLightPositionX;
    QLabel *labelLightPositionY;
    QDoubleSpinBox *doubleSpinBoxLightPositionY;
    QLabel *labelLightPositionZ;
    QDoubleSpinBox *doubleSpinBoxLightPositionZ;
    QWidget *widgetLightAmbient;
    QHBoxLayout *horizontalLayoutLightAmbient;
    QLabel *labelLightAmbient;
    QPushButton *pushButtonLightAmbientValue;
    QLabel *labelLightAmbientNote;
    QDoubleSpinBox *doubleSpinBoxLightAmbient;
    QSpacerItem *horizontalSpacerLightAmbient;
    QWidget *widgetLightDiffuse;
    QHBoxLayout *horizontalLayoutLightDiffuse;
    QLabel *labelLightDiffuse;
    QPushButton *pushButtonLightDiffuseValue;
    QLabel *labelLightDiffuseNote;
    QDoubleSpinBox *doubleSpinBoxLightDiffuse;
    QSpacerItem *horizontalSpacerLightDiffuse;
    QWidget *widgetLightSpecular;
    QHBoxLayout *horizontalLayoutLightSpecular;
    QLabel *labelLightSpecular;
    QPushButton *pushButtonLightSpecularValue;
    QLabel *labelLightSpecularNote;
    QDoubleSpinBox *doubleSpinBoxLightSpecular;
    QSpacerItem *horizontalSpacerLightSpecular;
    QWidget *widgetMaterialShininess;
    QHBoxLayout *horizontalLayoutMaterialShininess;
    QLabel *labelMaterialShininess;
    QSlider *horizontalSliderMaterialShininess;
    QLabel *labelMaterialShininessValue;
    QSpacerItem *verticalSpacerLeft3;
    QGroupBox *groupBoxConfigurations;
    QVBoxLayout *verticalLayoutConfigurations;
    QWidget *widgetLoadSave;
    QHBoxLayout *horizontalLayoutLoadSave;
    QLabel *labelUserConfig;
    QPushButton *pushButtonLoad;
    QPushButton *pushButtonSave;
    QSpacerItem *horizontalSpacerLoadSave;
    QWidget *widgetStartEnd;
    QHBoxLayout *horizontalLayout;
    QLabel *labelMarkPoint;
    QPushButton *pushButtonStart;
    QPushButton *pushButtonEnd;
    QSpacerItem *horizontalSpacerStartEnd;
    QWidget *widgetComputeTrace;
    QHBoxLayout *horizontalLayoutComputeTrace;
    QLabel *labelEntropyMap;
    QPushButton *pushButtonCompute;
    QSpacerItem *horizontalSpacerComputeTrace;
    QWidget *widgetAnimation;
    QHBoxLayout *horizontalLayoutAnimation;
    QLabel *labelAnimation;
    QPushButton *pushButtonTrace;
    QSpacerItem *horizontalSpacer;
    QWidget *widgetRight;
    QVBoxLayout *verticalLayoutRight;
    QGroupBox *groupBoxGaussianFilter;
    QVBoxLayout *verticalLayout_2;
    QWidget *widgetGaussian1D;
    QHBoxLayout *horizontalLayoutGaussian1D;
    QLabel *labelGaussian1D;
    QRadioButton *radioButtonGaussian1D1;
    QRadioButton *radioButtonGaussian1D3;
    QRadioButton *radioButtonGaussian1D5;
    QRadioButton *radioButtonGaussian1D7;
    QSpacerItem *horizontalSpacerGaussian1D;
    QWidget *widgetGaussian2D;
    QHBoxLayout *horizontalLayoutGaussian2D;
    QLabel *labelGaussian2D;
    QRadioButton *radioButtonGaussian2D1;
    QRadioButton *radioButtonGaussian2D3;
    QRadioButton *radioButtonGaussian2D5;
    QRadioButton *radioButtonGaussian2D7;
    QSpacerItem *horizontalSpacerGaussian2D;
    QGroupBox *groupBoxVisibility;
    QVBoxLayout *verticalLayoutVisibility;
    QWidget *widgetViewEntropy;
    QHBoxLayout *horizontalLayoutViewEntropy;
    QLabel *labelViewEntropy;
    QLabel *labelViewEntropyValue;
    QSpacerItem *horizontalSpacerViewEntropy;
    QWidget *widgetEntropyMap;
    QHBoxLayout *horizontalLayoutEntropyMap;
    QHemisphere *widgetNorthernHemisphere;
    QVBoxLayout *verticalLayoutNorthernHemisphere;
    QLabel *labelNorthernHemisphere;
    QSpacerItem *verticalSpacerNorthernHemisphere;
    QHemisphere *widgetSouthernHemisphere;
    QVBoxLayout *verticalLayoutSouthernHemisphere;
    QLabel *labelSouthernHemisphere;
    QSpacerItem *verticalSpacerSouthernHemisphere;
    QGroupBox *groupBoxEditor;
    QVBoxLayout *verticalLayoutEditor;
    QTransferFunction1D *widgetEditor;

    void setupUi(QWidget *QVSControlPanel)
    {
        if (QVSControlPanel->objectName().isEmpty())
            QVSControlPanel->setObjectName(QString::fromUtf8("QVSControlPanel"));
        QVSControlPanel->resize(786, 552);
        QVSControlPanel->setMinimumSize(QSize(0, 0));
        QVSControlPanel->setMaximumSize(QSize(16777215, 16777215));
        verticalLayout = new QVBoxLayout(QVSControlPanel);
#ifndef Q_OS_MAC
        verticalLayout->setSpacing(6);
#endif
        verticalLayout->setContentsMargins(3, 3, 3, 3);
        verticalLayout->setObjectName(QString::fromUtf8("verticalLayout"));
        widgetTop = new QWidget(QVSControlPanel);
        widgetTop->setObjectName(QString::fromUtf8("widgetTop"));
        horizontalLayoutTop = new QHBoxLayout(widgetTop);
#ifndef Q_OS_MAC
        horizontalLayoutTop->setSpacing(6);
#endif
        horizontalLayoutTop->setContentsMargins(3, 3, 3, 3);
        horizontalLayoutTop->setObjectName(QString::fromUtf8("horizontalLayoutTop"));
        widgetLeft = new QWidget(widgetTop);
        widgetLeft->setObjectName(QString::fromUtf8("widgetLeft"));
        widgetLeft->setMinimumSize(QSize(256, 540));
        widgetLeft->setMaximumSize(QSize(256, 16777215));
        verticalLayoutLeft = new QVBoxLayout(widgetLeft);
#ifndef Q_OS_MAC
        verticalLayoutLeft->setSpacing(6);
#endif
        verticalLayoutLeft->setContentsMargins(6, 6, 6, 6);
        verticalLayoutLeft->setObjectName(QString::fromUtf8("verticalLayoutLeft"));
        groupBoxRayCasting = new QGroupBox(widgetLeft);
        groupBoxRayCasting->setObjectName(QString::fromUtf8("groupBoxRayCasting"));
        groupBoxRayCasting->setMinimumSize(QSize(0, 40));
        groupBoxRayCasting->setMaximumSize(QSize(16777215, 40));
        groupBoxRayCasting->setCheckable(true);
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
        labelStepSize->setAlignment(Qt::AlignLeading|Qt::AlignLeft|Qt::AlignVCenter);

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


        verticalLayoutLeft->addWidget(groupBoxRayCasting);

        verticalSpacerLeft2 = new QSpacerItem(20, 40, QSizePolicy::Minimum, QSizePolicy::Expanding);

        verticalLayoutLeft->addItem(verticalSpacerLeft2);

        groupBoxRendering = new QGroupBox(widgetLeft);
        groupBoxRendering->setObjectName(QString::fromUtf8("groupBoxRendering"));
        groupBoxRendering->setMinimumSize(QSize(0, 80));
        groupBoxRendering->setMaximumSize(QSize(16777215, 80));
        groupBoxRendering->setCheckable(true);
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
        labelVolumeScale->setAlignment(Qt::AlignLeading|Qt::AlignLeft|Qt::AlignVCenter);

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
        labelVolumeOffset->setAlignment(Qt::AlignLeading|Qt::AlignLeft|Qt::AlignVCenter);

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


        verticalLayoutLeft->addWidget(groupBoxRendering);

        verticalSpacerLeft1 = new QSpacerItem(20, 40, QSizePolicy::Minimum, QSizePolicy::Expanding);

        verticalLayoutLeft->addItem(verticalSpacerLeft1);

        groupBoxIlluminationModel = new QGroupBox(widgetLeft);
        groupBoxIlluminationModel->setObjectName(QString::fromUtf8("groupBoxIlluminationModel"));
        groupBoxIlluminationModel->setMinimumSize(QSize(0, 200));
        groupBoxIlluminationModel->setMaximumSize(QSize(16777215, 200));
        groupBoxIlluminationModel->setCheckable(true);
        groupBoxIlluminationModel->setChecked(false);
        verticalLayoutIlluminationModel = new QVBoxLayout(groupBoxIlluminationModel);
        verticalLayoutIlluminationModel->setContentsMargins(6, 6, 6, 6);
        verticalLayoutIlluminationModel->setObjectName(QString::fromUtf8("verticalLayoutIlluminationModel"));
        widgetLightPosition = new QWidget(groupBoxIlluminationModel);
        widgetLightPosition->setObjectName(QString::fromUtf8("widgetLightPosition"));
        horizontalLayoutLightPosition = new QHBoxLayout(widgetLightPosition);
        horizontalLayoutLightPosition->setSpacing(3);
        horizontalLayoutLightPosition->setContentsMargins(0, 0, 0, 0);
        horizontalLayoutLightPosition->setObjectName(QString::fromUtf8("horizontalLayoutLightPosition"));
        labelLightPosition = new QLabel(widgetLightPosition);
        labelLightPosition->setObjectName(QString::fromUtf8("labelLightPosition"));
        labelLightPosition->setMinimumSize(QSize(40, 0));
        labelLightPosition->setMaximumSize(QSize(40, 16777215));

        horizontalLayoutLightPosition->addWidget(labelLightPosition);

        horizontalSpacerLightPosition = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayoutLightPosition->addItem(horizontalSpacerLightPosition);

        labelLightPositionX = new QLabel(widgetLightPosition);
        labelLightPositionX->setObjectName(QString::fromUtf8("labelLightPositionX"));
        labelLightPositionX->setMinimumSize(QSize(10, 0));
        labelLightPositionX->setMaximumSize(QSize(10, 16777215));
        labelLightPositionX->setAlignment(Qt::AlignRight|Qt::AlignTrailing|Qt::AlignVCenter);

        horizontalLayoutLightPosition->addWidget(labelLightPositionX);

        doubleSpinBoxLightPositionX = new QDoubleSpinBox(widgetLightPosition);
        doubleSpinBoxLightPositionX->setObjectName(QString::fromUtf8("doubleSpinBoxLightPositionX"));
        doubleSpinBoxLightPositionX->setMinimumSize(QSize(45, 0));
        doubleSpinBoxLightPositionX->setMaximumSize(QSize(45, 16777215));
        doubleSpinBoxLightPositionX->setDecimals(2);
        doubleSpinBoxLightPositionX->setMinimum(-9.99);
        doubleSpinBoxLightPositionX->setMaximum(9.99);
        doubleSpinBoxLightPositionX->setSingleStep(0.01);
        doubleSpinBoxLightPositionX->setValue(1);

        horizontalLayoutLightPosition->addWidget(doubleSpinBoxLightPositionX);

        labelLightPositionY = new QLabel(widgetLightPosition);
        labelLightPositionY->setObjectName(QString::fromUtf8("labelLightPositionY"));
        labelLightPositionY->setMinimumSize(QSize(10, 0));
        labelLightPositionY->setMaximumSize(QSize(10, 16777215));
        labelLightPositionY->setAlignment(Qt::AlignRight|Qt::AlignTrailing|Qt::AlignVCenter);

        horizontalLayoutLightPosition->addWidget(labelLightPositionY);

        doubleSpinBoxLightPositionY = new QDoubleSpinBox(widgetLightPosition);
        doubleSpinBoxLightPositionY->setObjectName(QString::fromUtf8("doubleSpinBoxLightPositionY"));
        doubleSpinBoxLightPositionY->setMinimumSize(QSize(45, 0));
        doubleSpinBoxLightPositionY->setMaximumSize(QSize(45, 16777215));
        doubleSpinBoxLightPositionY->setDecimals(2);
        doubleSpinBoxLightPositionY->setMinimum(-9.99);
        doubleSpinBoxLightPositionY->setMaximum(9.99);
        doubleSpinBoxLightPositionY->setSingleStep(0.01);
        doubleSpinBoxLightPositionY->setValue(1);

        horizontalLayoutLightPosition->addWidget(doubleSpinBoxLightPositionY);

        labelLightPositionZ = new QLabel(widgetLightPosition);
        labelLightPositionZ->setObjectName(QString::fromUtf8("labelLightPositionZ"));
        labelLightPositionZ->setMinimumSize(QSize(10, 0));
        labelLightPositionZ->setMaximumSize(QSize(10, 16777215));
        labelLightPositionZ->setAlignment(Qt::AlignRight|Qt::AlignTrailing|Qt::AlignVCenter);

        horizontalLayoutLightPosition->addWidget(labelLightPositionZ);

        doubleSpinBoxLightPositionZ = new QDoubleSpinBox(widgetLightPosition);
        doubleSpinBoxLightPositionZ->setObjectName(QString::fromUtf8("doubleSpinBoxLightPositionZ"));
        doubleSpinBoxLightPositionZ->setMinimumSize(QSize(45, 0));
        doubleSpinBoxLightPositionZ->setMaximumSize(QSize(45, 16777215));
        doubleSpinBoxLightPositionZ->setDecimals(2);
        doubleSpinBoxLightPositionZ->setMinimum(-9.99);
        doubleSpinBoxLightPositionZ->setMaximum(9.99);
        doubleSpinBoxLightPositionZ->setSingleStep(0.01);
        doubleSpinBoxLightPositionZ->setValue(1);

        horizontalLayoutLightPosition->addWidget(doubleSpinBoxLightPositionZ);


        verticalLayoutIlluminationModel->addWidget(widgetLightPosition);

        widgetLightAmbient = new QWidget(groupBoxIlluminationModel);
        widgetLightAmbient->setObjectName(QString::fromUtf8("widgetLightAmbient"));
        horizontalLayoutLightAmbient = new QHBoxLayout(widgetLightAmbient);
#ifndef Q_OS_MAC
        horizontalLayoutLightAmbient->setSpacing(6);
#endif
        horizontalLayoutLightAmbient->setContentsMargins(0, 0, 0, 0);
        horizontalLayoutLightAmbient->setObjectName(QString::fromUtf8("horizontalLayoutLightAmbient"));
        labelLightAmbient = new QLabel(widgetLightAmbient);
        labelLightAmbient->setObjectName(QString::fromUtf8("labelLightAmbient"));
        labelLightAmbient->setMinimumSize(QSize(96, 0));
        labelLightAmbient->setMaximumSize(QSize(96, 16777215));

        horizontalLayoutLightAmbient->addWidget(labelLightAmbient);

        pushButtonLightAmbientValue = new QPushButton(widgetLightAmbient);
        pushButtonLightAmbientValue->setObjectName(QString::fromUtf8("pushButtonLightAmbientValue"));
        pushButtonLightAmbientValue->setMinimumSize(QSize(30, 15));
        pushButtonLightAmbientValue->setMaximumSize(QSize(30, 15));
        pushButtonLightAmbientValue->setStyleSheet(QString::fromUtf8("border: 1px solid gray;\n"
"background-color: rgb(255, 255, 127);"));
        pushButtonLightAmbientValue->setFlat(true);

        horizontalLayoutLightAmbient->addWidget(pushButtonLightAmbientValue);

        labelLightAmbientNote = new QLabel(widgetLightAmbient);
        labelLightAmbientNote->setObjectName(QString::fromUtf8("labelLightAmbientNote"));
        labelLightAmbientNote->setMinimumSize(QSize(20, 0));
        labelLightAmbientNote->setMaximumSize(QSize(20, 16777215));
        labelLightAmbientNote->setAlignment(Qt::AlignRight|Qt::AlignTrailing|Qt::AlignVCenter);

        horizontalLayoutLightAmbient->addWidget(labelLightAmbientNote);

        doubleSpinBoxLightAmbient = new QDoubleSpinBox(widgetLightAmbient);
        doubleSpinBoxLightAmbient->setObjectName(QString::fromUtf8("doubleSpinBoxLightAmbient"));
        doubleSpinBoxLightAmbient->setMaximum(1);
        doubleSpinBoxLightAmbient->setSingleStep(0.01);
        doubleSpinBoxLightAmbient->setValue(0.8);

        horizontalLayoutLightAmbient->addWidget(doubleSpinBoxLightAmbient);

        horizontalSpacerLightAmbient = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayoutLightAmbient->addItem(horizontalSpacerLightAmbient);


        verticalLayoutIlluminationModel->addWidget(widgetLightAmbient);

        widgetLightDiffuse = new QWidget(groupBoxIlluminationModel);
        widgetLightDiffuse->setObjectName(QString::fromUtf8("widgetLightDiffuse"));
        horizontalLayoutLightDiffuse = new QHBoxLayout(widgetLightDiffuse);
        horizontalLayoutLightDiffuse->setContentsMargins(0, 0, 0, 0);
        horizontalLayoutLightDiffuse->setObjectName(QString::fromUtf8("horizontalLayoutLightDiffuse"));
        labelLightDiffuse = new QLabel(widgetLightDiffuse);
        labelLightDiffuse->setObjectName(QString::fromUtf8("labelLightDiffuse"));
        labelLightDiffuse->setMinimumSize(QSize(96, 0));
        labelLightDiffuse->setMaximumSize(QSize(96, 16777215));

        horizontalLayoutLightDiffuse->addWidget(labelLightDiffuse);

        pushButtonLightDiffuseValue = new QPushButton(widgetLightDiffuse);
        pushButtonLightDiffuseValue->setObjectName(QString::fromUtf8("pushButtonLightDiffuseValue"));
        pushButtonLightDiffuseValue->setMinimumSize(QSize(30, 15));
        pushButtonLightDiffuseValue->setMaximumSize(QSize(30, 15));
        pushButtonLightDiffuseValue->setStyleSheet(QString::fromUtf8("border: 1px solid gray;\n"
"background-color: rgb(64, 64, 64);"));
        pushButtonLightDiffuseValue->setFlat(true);

        horizontalLayoutLightDiffuse->addWidget(pushButtonLightDiffuseValue);

        labelLightDiffuseNote = new QLabel(widgetLightDiffuse);
        labelLightDiffuseNote->setObjectName(QString::fromUtf8("labelLightDiffuseNote"));
        labelLightDiffuseNote->setMinimumSize(QSize(20, 0));
        labelLightDiffuseNote->setMaximumSize(QSize(20, 16777215));
        labelLightDiffuseNote->setAlignment(Qt::AlignRight|Qt::AlignTrailing|Qt::AlignVCenter);

        horizontalLayoutLightDiffuse->addWidget(labelLightDiffuseNote);

        doubleSpinBoxLightDiffuse = new QDoubleSpinBox(widgetLightDiffuse);
        doubleSpinBoxLightDiffuse->setObjectName(QString::fromUtf8("doubleSpinBoxLightDiffuse"));
        doubleSpinBoxLightDiffuse->setMaximum(1);
        doubleSpinBoxLightDiffuse->setSingleStep(0.01);
        doubleSpinBoxLightDiffuse->setValue(0.3);

        horizontalLayoutLightDiffuse->addWidget(doubleSpinBoxLightDiffuse);

        horizontalSpacerLightDiffuse = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayoutLightDiffuse->addItem(horizontalSpacerLightDiffuse);


        verticalLayoutIlluminationModel->addWidget(widgetLightDiffuse);

        widgetLightSpecular = new QWidget(groupBoxIlluminationModel);
        widgetLightSpecular->setObjectName(QString::fromUtf8("widgetLightSpecular"));
        horizontalLayoutLightSpecular = new QHBoxLayout(widgetLightSpecular);
        horizontalLayoutLightSpecular->setContentsMargins(0, 0, 0, 0);
        horizontalLayoutLightSpecular->setObjectName(QString::fromUtf8("horizontalLayoutLightSpecular"));
        labelLightSpecular = new QLabel(widgetLightSpecular);
        labelLightSpecular->setObjectName(QString::fromUtf8("labelLightSpecular"));
        labelLightSpecular->setMinimumSize(QSize(96, 0));
        labelLightSpecular->setMaximumSize(QSize(96, 16777215));

        horizontalLayoutLightSpecular->addWidget(labelLightSpecular);

        pushButtonLightSpecularValue = new QPushButton(widgetLightSpecular);
        pushButtonLightSpecularValue->setObjectName(QString::fromUtf8("pushButtonLightSpecularValue"));
        pushButtonLightSpecularValue->setMinimumSize(QSize(30, 15));
        pushButtonLightSpecularValue->setMaximumSize(QSize(30, 15));
        pushButtonLightSpecularValue->setStyleSheet(QString::fromUtf8("border: 1px solid gray;\n"
"background-color: rgb(255, 255, 255);"));
        pushButtonLightSpecularValue->setFlat(true);

        horizontalLayoutLightSpecular->addWidget(pushButtonLightSpecularValue);

        labelLightSpecularNote = new QLabel(widgetLightSpecular);
        labelLightSpecularNote->setObjectName(QString::fromUtf8("labelLightSpecularNote"));
        labelLightSpecularNote->setMinimumSize(QSize(20, 0));
        labelLightSpecularNote->setMaximumSize(QSize(20, 16777215));
        labelLightSpecularNote->setAlignment(Qt::AlignRight|Qt::AlignTrailing|Qt::AlignVCenter);

        horizontalLayoutLightSpecular->addWidget(labelLightSpecularNote);

        doubleSpinBoxLightSpecular = new QDoubleSpinBox(widgetLightSpecular);
        doubleSpinBoxLightSpecular->setObjectName(QString::fromUtf8("doubleSpinBoxLightSpecular"));
        doubleSpinBoxLightSpecular->setMaximum(1);
        doubleSpinBoxLightSpecular->setSingleStep(0.01);
        doubleSpinBoxLightSpecular->setValue(0.1);

        horizontalLayoutLightSpecular->addWidget(doubleSpinBoxLightSpecular);

        horizontalSpacerLightSpecular = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayoutLightSpecular->addItem(horizontalSpacerLightSpecular);


        verticalLayoutIlluminationModel->addWidget(widgetLightSpecular);

        widgetMaterialShininess = new QWidget(groupBoxIlluminationModel);
        widgetMaterialShininess->setObjectName(QString::fromUtf8("widgetMaterialShininess"));
        widgetMaterialShininess->setMinimumSize(QSize(0, 0));
        widgetMaterialShininess->setMaximumSize(QSize(16777215, 16777215));
        horizontalLayoutMaterialShininess = new QHBoxLayout(widgetMaterialShininess);
        horizontalLayoutMaterialShininess->setSpacing(3);
        horizontalLayoutMaterialShininess->setContentsMargins(0, 0, 0, 0);
        horizontalLayoutMaterialShininess->setObjectName(QString::fromUtf8("horizontalLayoutMaterialShininess"));
        labelMaterialShininess = new QLabel(widgetMaterialShininess);
        labelMaterialShininess->setObjectName(QString::fromUtf8("labelMaterialShininess"));
        labelMaterialShininess->setMinimumSize(QSize(96, 0));
        labelMaterialShininess->setMaximumSize(QSize(96, 16777215));

        horizontalLayoutMaterialShininess->addWidget(labelMaterialShininess);

        horizontalSliderMaterialShininess = new QSlider(widgetMaterialShininess);
        horizontalSliderMaterialShininess->setObjectName(QString::fromUtf8("horizontalSliderMaterialShininess"));
        horizontalSliderMaterialShininess->setMinimum(1);
        horizontalSliderMaterialShininess->setMaximum(100);
        horizontalSliderMaterialShininess->setOrientation(Qt::Horizontal);

        horizontalLayoutMaterialShininess->addWidget(horizontalSliderMaterialShininess);

        labelMaterialShininessValue = new QLabel(widgetMaterialShininess);
        labelMaterialShininessValue->setObjectName(QString::fromUtf8("labelMaterialShininessValue"));
        labelMaterialShininessValue->setMinimumSize(QSize(32, 0));
        labelMaterialShininessValue->setMaximumSize(QSize(32, 16777215));
        labelMaterialShininessValue->setAlignment(Qt::AlignRight|Qt::AlignTrailing|Qt::AlignVCenter);

        horizontalLayoutMaterialShininess->addWidget(labelMaterialShininessValue);


        verticalLayoutIlluminationModel->addWidget(widgetMaterialShininess);


        verticalLayoutLeft->addWidget(groupBoxIlluminationModel);

        verticalSpacerLeft3 = new QSpacerItem(20, 40, QSizePolicy::Minimum, QSizePolicy::Expanding);

        verticalLayoutLeft->addItem(verticalSpacerLeft3);

        groupBoxConfigurations = new QGroupBox(widgetLeft);
        groupBoxConfigurations->setObjectName(QString::fromUtf8("groupBoxConfigurations"));
        groupBoxConfigurations->setMinimumSize(QSize(0, 160));
        groupBoxConfigurations->setMaximumSize(QSize(16777215, 160));
        groupBoxConfigurations->setCheckable(true);
        verticalLayoutConfigurations = new QVBoxLayout(groupBoxConfigurations);
#ifndef Q_OS_MAC
        verticalLayoutConfigurations->setSpacing(6);
#endif
        verticalLayoutConfigurations->setContentsMargins(6, 6, 6, 6);
        verticalLayoutConfigurations->setObjectName(QString::fromUtf8("verticalLayoutConfigurations"));
        widgetLoadSave = new QWidget(groupBoxConfigurations);
        widgetLoadSave->setObjectName(QString::fromUtf8("widgetLoadSave"));
        widgetLoadSave->setMinimumSize(QSize(0, 30));
        widgetLoadSave->setMaximumSize(QSize(16777215, 30));
        horizontalLayoutLoadSave = new QHBoxLayout(widgetLoadSave);
        horizontalLayoutLoadSave->setSpacing(3);
        horizontalLayoutLoadSave->setContentsMargins(0, 0, 0, 0);
        horizontalLayoutLoadSave->setObjectName(QString::fromUtf8("horizontalLayoutLoadSave"));
        labelUserConfig = new QLabel(widgetLoadSave);
        labelUserConfig->setObjectName(QString::fromUtf8("labelUserConfig"));
        labelUserConfig->setMinimumSize(QSize(96, 0));
        labelUserConfig->setMaximumSize(QSize(96, 16777215));
        labelUserConfig->setAlignment(Qt::AlignLeading|Qt::AlignLeft|Qt::AlignVCenter);

        horizontalLayoutLoadSave->addWidget(labelUserConfig);

        pushButtonLoad = new QPushButton(widgetLoadSave);
        pushButtonLoad->setObjectName(QString::fromUtf8("pushButtonLoad"));
        pushButtonLoad->setMinimumSize(QSize(60, 0));
        pushButtonLoad->setMaximumSize(QSize(60, 16777215));

        horizontalLayoutLoadSave->addWidget(pushButtonLoad);

        pushButtonSave = new QPushButton(widgetLoadSave);
        pushButtonSave->setObjectName(QString::fromUtf8("pushButtonSave"));
        pushButtonSave->setMinimumSize(QSize(60, 0));
        pushButtonSave->setMaximumSize(QSize(60, 16777215));

        horizontalLayoutLoadSave->addWidget(pushButtonSave);

        horizontalSpacerLoadSave = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayoutLoadSave->addItem(horizontalSpacerLoadSave);

        pushButtonLoad->raise();
        pushButtonSave->raise();
        labelUserConfig->raise();

        verticalLayoutConfigurations->addWidget(widgetLoadSave);

        widgetStartEnd = new QWidget(groupBoxConfigurations);
        widgetStartEnd->setObjectName(QString::fromUtf8("widgetStartEnd"));
        widgetStartEnd->setMinimumSize(QSize(0, 30));
        widgetStartEnd->setMaximumSize(QSize(16777215, 30));
        horizontalLayout = new QHBoxLayout(widgetStartEnd);
        horizontalLayout->setSpacing(3);
        horizontalLayout->setContentsMargins(0, 0, 0, 0);
        horizontalLayout->setObjectName(QString::fromUtf8("horizontalLayout"));
        labelMarkPoint = new QLabel(widgetStartEnd);
        labelMarkPoint->setObjectName(QString::fromUtf8("labelMarkPoint"));
        labelMarkPoint->setMinimumSize(QSize(96, 0));
        labelMarkPoint->setMaximumSize(QSize(96, 16777215));
        labelMarkPoint->setAlignment(Qt::AlignLeading|Qt::AlignLeft|Qt::AlignVCenter);

        horizontalLayout->addWidget(labelMarkPoint);

        pushButtonStart = new QPushButton(widgetStartEnd);
        pushButtonStart->setObjectName(QString::fromUtf8("pushButtonStart"));
        pushButtonStart->setMinimumSize(QSize(60, 0));
        pushButtonStart->setMaximumSize(QSize(60, 16777215));

        horizontalLayout->addWidget(pushButtonStart);

        pushButtonEnd = new QPushButton(widgetStartEnd);
        pushButtonEnd->setObjectName(QString::fromUtf8("pushButtonEnd"));
        pushButtonEnd->setMinimumSize(QSize(60, 0));
        pushButtonEnd->setMaximumSize(QSize(60, 16777215));

        horizontalLayout->addWidget(pushButtonEnd);

        horizontalSpacerStartEnd = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout->addItem(horizontalSpacerStartEnd);


        verticalLayoutConfigurations->addWidget(widgetStartEnd);

        widgetComputeTrace = new QWidget(groupBoxConfigurations);
        widgetComputeTrace->setObjectName(QString::fromUtf8("widgetComputeTrace"));
        widgetComputeTrace->setMinimumSize(QSize(0, 30));
        widgetComputeTrace->setMaximumSize(QSize(16777215, 30));
        horizontalLayoutComputeTrace = new QHBoxLayout(widgetComputeTrace);
        horizontalLayoutComputeTrace->setSpacing(3);
        horizontalLayoutComputeTrace->setContentsMargins(0, 0, 0, 0);
        horizontalLayoutComputeTrace->setObjectName(QString::fromUtf8("horizontalLayoutComputeTrace"));
        labelEntropyMap = new QLabel(widgetComputeTrace);
        labelEntropyMap->setObjectName(QString::fromUtf8("labelEntropyMap"));
        labelEntropyMap->setMinimumSize(QSize(96, 0));
        labelEntropyMap->setMaximumSize(QSize(96, 16777215));
        labelEntropyMap->setAlignment(Qt::AlignLeading|Qt::AlignLeft|Qt::AlignVCenter);

        horizontalLayoutComputeTrace->addWidget(labelEntropyMap);

        pushButtonCompute = new QPushButton(widgetComputeTrace);
        pushButtonCompute->setObjectName(QString::fromUtf8("pushButtonCompute"));
        pushButtonCompute->setMinimumSize(QSize(60, 0));
        pushButtonCompute->setMaximumSize(QSize(60, 16777215));

        horizontalLayoutComputeTrace->addWidget(pushButtonCompute);

        horizontalSpacerComputeTrace = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayoutComputeTrace->addItem(horizontalSpacerComputeTrace);


        verticalLayoutConfigurations->addWidget(widgetComputeTrace);

        widgetAnimation = new QWidget(groupBoxConfigurations);
        widgetAnimation->setObjectName(QString::fromUtf8("widgetAnimation"));
        widgetAnimation->setMinimumSize(QSize(0, 30));
        widgetAnimation->setMaximumSize(QSize(16777215, 30));
        horizontalLayoutAnimation = new QHBoxLayout(widgetAnimation);
        horizontalLayoutAnimation->setSpacing(3);
        horizontalLayoutAnimation->setContentsMargins(0, 0, 0, 0);
        horizontalLayoutAnimation->setObjectName(QString::fromUtf8("horizontalLayoutAnimation"));
        labelAnimation = new QLabel(widgetAnimation);
        labelAnimation->setObjectName(QString::fromUtf8("labelAnimation"));
        labelAnimation->setMinimumSize(QSize(96, 0));
        labelAnimation->setMaximumSize(QSize(96, 16777215));
        labelAnimation->setAlignment(Qt::AlignLeading|Qt::AlignLeft|Qt::AlignVCenter);

        horizontalLayoutAnimation->addWidget(labelAnimation);

        pushButtonTrace = new QPushButton(widgetAnimation);
        pushButtonTrace->setObjectName(QString::fromUtf8("pushButtonTrace"));
        pushButtonTrace->setMinimumSize(QSize(60, 0));
        pushButtonTrace->setMaximumSize(QSize(60, 16777215));

        horizontalLayoutAnimation->addWidget(pushButtonTrace);

        horizontalSpacer = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayoutAnimation->addItem(horizontalSpacer);


        verticalLayoutConfigurations->addWidget(widgetAnimation);


        verticalLayoutLeft->addWidget(groupBoxConfigurations);


        horizontalLayoutTop->addWidget(widgetLeft);

        widgetRight = new QWidget(widgetTop);
        widgetRight->setObjectName(QString::fromUtf8("widgetRight"));
        widgetRight->setMinimumSize(QSize(512, 0));
        verticalLayoutRight = new QVBoxLayout(widgetRight);
#ifndef Q_OS_MAC
        verticalLayoutRight->setSpacing(6);
#endif
        verticalLayoutRight->setContentsMargins(6, 6, 6, 6);
        verticalLayoutRight->setObjectName(QString::fromUtf8("verticalLayoutRight"));
        groupBoxGaussianFilter = new QGroupBox(widgetRight);
        groupBoxGaussianFilter->setObjectName(QString::fromUtf8("groupBoxGaussianFilter"));
        groupBoxGaussianFilter->setMinimumSize(QSize(0, 80));
        groupBoxGaussianFilter->setMaximumSize(QSize(16777215, 80));
        groupBoxGaussianFilter->setCheckable(true);
        verticalLayout_2 = new QVBoxLayout(groupBoxGaussianFilter);
#ifndef Q_OS_MAC
        verticalLayout_2->setSpacing(6);
#endif
        verticalLayout_2->setContentsMargins(6, 6, 6, 6);
        verticalLayout_2->setObjectName(QString::fromUtf8("verticalLayout_2"));
        widgetGaussian1D = new QWidget(groupBoxGaussianFilter);
        widgetGaussian1D->setObjectName(QString::fromUtf8("widgetGaussian1D"));
        horizontalLayoutGaussian1D = new QHBoxLayout(widgetGaussian1D);
        horizontalLayoutGaussian1D->setSpacing(0);
        horizontalLayoutGaussian1D->setContentsMargins(0, 0, 0, 0);
        horizontalLayoutGaussian1D->setObjectName(QString::fromUtf8("horizontalLayoutGaussian1D"));
        labelGaussian1D = new QLabel(widgetGaussian1D);
        labelGaussian1D->setObjectName(QString::fromUtf8("labelGaussian1D"));
        labelGaussian1D->setMinimumSize(QSize(128, 0));
        labelGaussian1D->setMaximumSize(QSize(128, 16777215));

        horizontalLayoutGaussian1D->addWidget(labelGaussian1D);

        radioButtonGaussian1D1 = new QRadioButton(widgetGaussian1D);
        radioButtonGaussian1D1->setObjectName(QString::fromUtf8("radioButtonGaussian1D1"));
        radioButtonGaussian1D1->setMinimumSize(QSize(48, 0));
        radioButtonGaussian1D1->setMaximumSize(QSize(48, 16777215));

        horizontalLayoutGaussian1D->addWidget(radioButtonGaussian1D1);

        radioButtonGaussian1D3 = new QRadioButton(widgetGaussian1D);
        radioButtonGaussian1D3->setObjectName(QString::fromUtf8("radioButtonGaussian1D3"));
        radioButtonGaussian1D3->setMinimumSize(QSize(48, 0));
        radioButtonGaussian1D3->setMaximumSize(QSize(48, 16777215));
        radioButtonGaussian1D3->setChecked(true);

        horizontalLayoutGaussian1D->addWidget(radioButtonGaussian1D3);

        radioButtonGaussian1D5 = new QRadioButton(widgetGaussian1D);
        radioButtonGaussian1D5->setObjectName(QString::fromUtf8("radioButtonGaussian1D5"));
        radioButtonGaussian1D5->setMinimumSize(QSize(48, 0));
        radioButtonGaussian1D5->setMaximumSize(QSize(48, 16777215));

        horizontalLayoutGaussian1D->addWidget(radioButtonGaussian1D5);

        radioButtonGaussian1D7 = new QRadioButton(widgetGaussian1D);
        radioButtonGaussian1D7->setObjectName(QString::fromUtf8("radioButtonGaussian1D7"));
        radioButtonGaussian1D7->setMinimumSize(QSize(48, 0));
        radioButtonGaussian1D7->setMaximumSize(QSize(48, 16777215));

        horizontalLayoutGaussian1D->addWidget(radioButtonGaussian1D7);

        horizontalSpacerGaussian1D = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayoutGaussian1D->addItem(horizontalSpacerGaussian1D);


        verticalLayout_2->addWidget(widgetGaussian1D);

        widgetGaussian2D = new QWidget(groupBoxGaussianFilter);
        widgetGaussian2D->setObjectName(QString::fromUtf8("widgetGaussian2D"));
        horizontalLayoutGaussian2D = new QHBoxLayout(widgetGaussian2D);
        horizontalLayoutGaussian2D->setSpacing(0);
        horizontalLayoutGaussian2D->setContentsMargins(0, 0, 0, 0);
        horizontalLayoutGaussian2D->setObjectName(QString::fromUtf8("horizontalLayoutGaussian2D"));
        labelGaussian2D = new QLabel(widgetGaussian2D);
        labelGaussian2D->setObjectName(QString::fromUtf8("labelGaussian2D"));
        labelGaussian2D->setMinimumSize(QSize(128, 0));
        labelGaussian2D->setMaximumSize(QSize(128, 16777215));

        horizontalLayoutGaussian2D->addWidget(labelGaussian2D);

        radioButtonGaussian2D1 = new QRadioButton(widgetGaussian2D);
        radioButtonGaussian2D1->setObjectName(QString::fromUtf8("radioButtonGaussian2D1"));
        radioButtonGaussian2D1->setMinimumSize(QSize(48, 0));
        radioButtonGaussian2D1->setMaximumSize(QSize(48, 16777215));

        horizontalLayoutGaussian2D->addWidget(radioButtonGaussian2D1);

        radioButtonGaussian2D3 = new QRadioButton(widgetGaussian2D);
        radioButtonGaussian2D3->setObjectName(QString::fromUtf8("radioButtonGaussian2D3"));
        radioButtonGaussian2D3->setMinimumSize(QSize(48, 0));
        radioButtonGaussian2D3->setMaximumSize(QSize(48, 16777215));
        radioButtonGaussian2D3->setChecked(true);

        horizontalLayoutGaussian2D->addWidget(radioButtonGaussian2D3);

        radioButtonGaussian2D5 = new QRadioButton(widgetGaussian2D);
        radioButtonGaussian2D5->setObjectName(QString::fromUtf8("radioButtonGaussian2D5"));
        radioButtonGaussian2D5->setMinimumSize(QSize(48, 0));
        radioButtonGaussian2D5->setMaximumSize(QSize(48, 16777215));

        horizontalLayoutGaussian2D->addWidget(radioButtonGaussian2D5);

        radioButtonGaussian2D7 = new QRadioButton(widgetGaussian2D);
        radioButtonGaussian2D7->setObjectName(QString::fromUtf8("radioButtonGaussian2D7"));
        radioButtonGaussian2D7->setMinimumSize(QSize(48, 0));
        radioButtonGaussian2D7->setMaximumSize(QSize(48, 16777215));

        horizontalLayoutGaussian2D->addWidget(radioButtonGaussian2D7);

        horizontalSpacerGaussian2D = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayoutGaussian2D->addItem(horizontalSpacerGaussian2D);


        verticalLayout_2->addWidget(widgetGaussian2D);


        verticalLayoutRight->addWidget(groupBoxGaussianFilter);

        groupBoxVisibility = new QGroupBox(widgetRight);
        groupBoxVisibility->setObjectName(QString::fromUtf8("groupBoxVisibility"));
        groupBoxVisibility->setMinimumSize(QSize(0, 250));
        groupBoxVisibility->setMaximumSize(QSize(16777215, 250));
        groupBoxVisibility->setCheckable(true);
        groupBoxVisibility->setChecked(false);
        verticalLayoutVisibility = new QVBoxLayout(groupBoxVisibility);
        verticalLayoutVisibility->setSpacing(0);
        verticalLayoutVisibility->setContentsMargins(0, 0, 0, 0);
        verticalLayoutVisibility->setObjectName(QString::fromUtf8("verticalLayoutVisibility"));
        widgetViewEntropy = new QWidget(groupBoxVisibility);
        widgetViewEntropy->setObjectName(QString::fromUtf8("widgetViewEntropy"));
        widgetViewEntropy->setMinimumSize(QSize(0, 20));
        widgetViewEntropy->setMaximumSize(QSize(16777215, 20));
        horizontalLayoutViewEntropy = new QHBoxLayout(widgetViewEntropy);
        horizontalLayoutViewEntropy->setSpacing(0);
        horizontalLayoutViewEntropy->setContentsMargins(0, 0, 0, 0);
        horizontalLayoutViewEntropy->setObjectName(QString::fromUtf8("horizontalLayoutViewEntropy"));
        labelViewEntropy = new QLabel(widgetViewEntropy);
        labelViewEntropy->setObjectName(QString::fromUtf8("labelViewEntropy"));
        labelViewEntropy->setMinimumSize(QSize(96, 0));
        labelViewEntropy->setMaximumSize(QSize(96, 16777215));

        horizontalLayoutViewEntropy->addWidget(labelViewEntropy);

        labelViewEntropyValue = new QLabel(widgetViewEntropy);
        labelViewEntropyValue->setObjectName(QString::fromUtf8("labelViewEntropyValue"));
        labelViewEntropyValue->setMinimumSize(QSize(100, 0));
        labelViewEntropyValue->setMaximumSize(QSize(100, 16777215));
        labelViewEntropyValue->setAlignment(Qt::AlignRight|Qt::AlignTrailing|Qt::AlignVCenter);

        horizontalLayoutViewEntropy->addWidget(labelViewEntropyValue);

        horizontalSpacerViewEntropy = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayoutViewEntropy->addItem(horizontalSpacerViewEntropy);


        verticalLayoutVisibility->addWidget(widgetViewEntropy);

        widgetEntropyMap = new QWidget(groupBoxVisibility);
        widgetEntropyMap->setObjectName(QString::fromUtf8("widgetEntropyMap"));
        widgetEntropyMap->setMinimumSize(QSize(0, 200));
        widgetEntropyMap->setMaximumSize(QSize(16777215, 200));
        horizontalLayoutEntropyMap = new QHBoxLayout(widgetEntropyMap);
        horizontalLayoutEntropyMap->setSpacing(0);
        horizontalLayoutEntropyMap->setContentsMargins(0, 0, 0, 0);
        horizontalLayoutEntropyMap->setObjectName(QString::fromUtf8("horizontalLayoutEntropyMap"));
        widgetNorthernHemisphere = new QHemisphere(widgetEntropyMap);
        widgetNorthernHemisphere->setObjectName(QString::fromUtf8("widgetNorthernHemisphere"));
        widgetNorthernHemisphere->setMinimumSize(QSize(200, 200));
        widgetNorthernHemisphere->setMaximumSize(QSize(200, 200));
        verticalLayoutNorthernHemisphere = new QVBoxLayout(widgetNorthernHemisphere);
        verticalLayoutNorthernHemisphere->setSpacing(0);
        verticalLayoutNorthernHemisphere->setContentsMargins(0, 0, 0, 0);
        verticalLayoutNorthernHemisphere->setObjectName(QString::fromUtf8("verticalLayoutNorthernHemisphere"));
        labelNorthernHemisphere = new QLabel(widgetNorthernHemisphere);
        labelNorthernHemisphere->setObjectName(QString::fromUtf8("labelNorthernHemisphere"));
        labelNorthernHemisphere->setMinimumSize(QSize(0, 16));
        labelNorthernHemisphere->setMaximumSize(QSize(16777215, 16));
        labelNorthernHemisphere->setAlignment(Qt::AlignLeading|Qt::AlignLeft|Qt::AlignTop);

        verticalLayoutNorthernHemisphere->addWidget(labelNorthernHemisphere);

        verticalSpacerNorthernHemisphere = new QSpacerItem(20, 40, QSizePolicy::Minimum, QSizePolicy::Expanding);

        verticalLayoutNorthernHemisphere->addItem(verticalSpacerNorthernHemisphere);


        horizontalLayoutEntropyMap->addWidget(widgetNorthernHemisphere);

        widgetSouthernHemisphere = new QHemisphere(widgetEntropyMap);
        widgetSouthernHemisphere->setObjectName(QString::fromUtf8("widgetSouthernHemisphere"));
        widgetSouthernHemisphere->setMinimumSize(QSize(200, 200));
        widgetSouthernHemisphere->setMaximumSize(QSize(200, 200));
        verticalLayoutSouthernHemisphere = new QVBoxLayout(widgetSouthernHemisphere);
        verticalLayoutSouthernHemisphere->setSpacing(0);
        verticalLayoutSouthernHemisphere->setContentsMargins(0, 0, 0, 0);
        verticalLayoutSouthernHemisphere->setObjectName(QString::fromUtf8("verticalLayoutSouthernHemisphere"));
        labelSouthernHemisphere = new QLabel(widgetSouthernHemisphere);
        labelSouthernHemisphere->setObjectName(QString::fromUtf8("labelSouthernHemisphere"));
        labelSouthernHemisphere->setMinimumSize(QSize(0, 16));
        labelSouthernHemisphere->setMaximumSize(QSize(16777215, 16));
        labelSouthernHemisphere->setAlignment(Qt::AlignLeading|Qt::AlignLeft|Qt::AlignTop);

        verticalLayoutSouthernHemisphere->addWidget(labelSouthernHemisphere);

        verticalSpacerSouthernHemisphere = new QSpacerItem(20, 40, QSizePolicy::Minimum, QSizePolicy::Expanding);

        verticalLayoutSouthernHemisphere->addItem(verticalSpacerSouthernHemisphere);


        horizontalLayoutEntropyMap->addWidget(widgetSouthernHemisphere);


        verticalLayoutVisibility->addWidget(widgetEntropyMap);


        verticalLayoutRight->addWidget(groupBoxVisibility);

        groupBoxEditor = new QGroupBox(widgetRight);
        groupBoxEditor->setObjectName(QString::fromUtf8("groupBoxEditor"));
        groupBoxEditor->setMinimumSize(QSize(0, 0));
        groupBoxEditor->setMaximumSize(QSize(16777215, 16777215));
        verticalLayoutEditor = new QVBoxLayout(groupBoxEditor);
        verticalLayoutEditor->setSpacing(0);
        verticalLayoutEditor->setContentsMargins(0, 0, 0, 0);
        verticalLayoutEditor->setObjectName(QString::fromUtf8("verticalLayoutEditor"));
        widgetEditor = new QTransferFunction1D(groupBoxEditor);
        widgetEditor->setObjectName(QString::fromUtf8("widgetEditor"));
        widgetEditor->setMinimumSize(QSize(256, 0));

        verticalLayoutEditor->addWidget(widgetEditor);


        verticalLayoutRight->addWidget(groupBoxEditor);


        horizontalLayoutTop->addWidget(widgetRight);


        verticalLayout->addWidget(widgetTop);


        retranslateUi(QVSControlPanel);
        QObject::connect(horizontalSliderStepSize, SIGNAL(valueChanged(int)), labelStepSizeValue, SLOT(setNum(int)));
        QObject::connect(horizontalSliderVolumeOffset, SIGNAL(valueChanged(int)), labelVolumeOffsetValue, SLOT(setNum(int)));
        QObject::connect(horizontalSliderVolumeScale, SIGNAL(valueChanged(int)), labelVolumeScaleValue, SLOT(setNum(int)));
        QObject::connect(horizontalSliderMaterialShininess, SIGNAL(valueChanged(int)), labelMaterialShininessValue, SLOT(setNum(int)));

        QMetaObject::connectSlotsByName(QVSControlPanel);
    } // setupUi

    void retranslateUi(QWidget *QVSControlPanel)
    {
        QVSControlPanel->setWindowTitle(QApplication::translate("QVSControlPanel", "Transfer Function Editor", 0, QApplication::UnicodeUTF8));
        groupBoxRayCasting->setTitle(QApplication::translate("QVSControlPanel", "1. Ray-casting", 0, QApplication::UnicodeUTF8));
        labelStepSize->setText(QApplication::translate("QVSControlPanel", "a.Step Size", 0, QApplication::UnicodeUTF8));
        labelStepSizeValue->setText(QApplication::translate("QVSControlPanel", "256", 0, QApplication::UnicodeUTF8));
        groupBoxRendering->setTitle(QApplication::translate("QVSControlPanel", "2. Volumetric Data", 0, QApplication::UnicodeUTF8));
        labelVolumeScale->setText(QApplication::translate("QVSControlPanel", "a.Volume Scale", 0, QApplication::UnicodeUTF8));
        labelVolumeScaleValue->setText(QApplication::translate("QVSControlPanel", "100", 0, QApplication::UnicodeUTF8));
        labelVolumeOffset->setText(QApplication::translate("QVSControlPanel", "b.Volume Offset", 0, QApplication::UnicodeUTF8));
        labelVolumeOffsetValue->setText(QApplication::translate("QVSControlPanel", "0", 0, QApplication::UnicodeUTF8));
        groupBoxIlluminationModel->setTitle(QApplication::translate("QVSControlPanel", "3. Illumination Model", 0, QApplication::UnicodeUTF8));
        labelLightPosition->setText(QApplication::translate("QVSControlPanel", "a.Light", 0, QApplication::UnicodeUTF8));
        labelLightPositionX->setText(QApplication::translate("QVSControlPanel", "x", 0, QApplication::UnicodeUTF8));
        labelLightPositionY->setText(QApplication::translate("QVSControlPanel", "y", 0, QApplication::UnicodeUTF8));
        labelLightPositionZ->setText(QApplication::translate("QVSControlPanel", "z", 0, QApplication::UnicodeUTF8));
        labelLightAmbient->setText(QApplication::translate("QVSControlPanel", "b.Light Ambient", 0, QApplication::UnicodeUTF8));
        pushButtonLightAmbientValue->setText(QString());
        labelLightAmbientNote->setText(QApplication::translate("QVSControlPanel", "ka", 0, QApplication::UnicodeUTF8));
        labelLightDiffuse->setText(QApplication::translate("QVSControlPanel", "c.Light Diffuse", 0, QApplication::UnicodeUTF8));
        pushButtonLightDiffuseValue->setText(QString());
        labelLightDiffuseNote->setText(QApplication::translate("QVSControlPanel", "kd", 0, QApplication::UnicodeUTF8));
        labelLightSpecular->setText(QApplication::translate("QVSControlPanel", "d.Light Specular", 0, QApplication::UnicodeUTF8));
        pushButtonLightSpecularValue->setText(QString());
        labelLightSpecularNote->setText(QApplication::translate("QVSControlPanel", "ks", 0, QApplication::UnicodeUTF8));
        labelMaterialShininess->setText(QApplication::translate("QVSControlPanel", "e.Shininess", 0, QApplication::UnicodeUTF8));
        labelMaterialShininessValue->setText(QApplication::translate("QVSControlPanel", "2", 0, QApplication::UnicodeUTF8));
        groupBoxConfigurations->setTitle(QApplication::translate("QVSControlPanel", "4. Configurations", 0, QApplication::UnicodeUTF8));
        labelUserConfig->setText(QApplication::translate("QVSControlPanel", "a.User Config", 0, QApplication::UnicodeUTF8));
        pushButtonLoad->setText(QApplication::translate("QVSControlPanel", "Load", 0, QApplication::UnicodeUTF8));
        pushButtonSave->setText(QApplication::translate("QVSControlPanel", "Save", 0, QApplication::UnicodeUTF8));
        labelMarkPoint->setText(QApplication::translate("QVSControlPanel", "b.Mark Point", 0, QApplication::UnicodeUTF8));
        pushButtonStart->setText(QApplication::translate("QVSControlPanel", "Start", 0, QApplication::UnicodeUTF8));
        pushButtonEnd->setText(QApplication::translate("QVSControlPanel", "End", 0, QApplication::UnicodeUTF8));
        labelEntropyMap->setText(QApplication::translate("QVSControlPanel", "c.Entropy Map", 0, QApplication::UnicodeUTF8));
        pushButtonCompute->setText(QApplication::translate("QVSControlPanel", "Compute", 0, QApplication::UnicodeUTF8));
        labelAnimation->setText(QApplication::translate("QVSControlPanel", "d.Animation", 0, QApplication::UnicodeUTF8));
        pushButtonTrace->setText(QApplication::translate("QVSControlPanel", "Trace", 0, QApplication::UnicodeUTF8));
        groupBoxGaussianFilter->setTitle(QApplication::translate("QVSControlPanel", "5. Gaussian filter", 0, QApplication::UnicodeUTF8));
        labelGaussian1D->setText(QApplication::translate("QVSControlPanel", "a.Gaussian Size 1D", 0, QApplication::UnicodeUTF8));
        radioButtonGaussian1D1->setText(QApplication::translate("QVSControlPanel", "1", 0, QApplication::UnicodeUTF8));
        radioButtonGaussian1D3->setText(QApplication::translate("QVSControlPanel", "3", 0, QApplication::UnicodeUTF8));
        radioButtonGaussian1D5->setText(QApplication::translate("QVSControlPanel", "5", 0, QApplication::UnicodeUTF8));
        radioButtonGaussian1D7->setText(QApplication::translate("QVSControlPanel", "7", 0, QApplication::UnicodeUTF8));
        labelGaussian2D->setText(QApplication::translate("QVSControlPanel", "b.Gaussian Size 2D ", 0, QApplication::UnicodeUTF8));
        radioButtonGaussian2D1->setText(QApplication::translate("QVSControlPanel", "1", 0, QApplication::UnicodeUTF8));
        radioButtonGaussian2D3->setText(QApplication::translate("QVSControlPanel", "3", 0, QApplication::UnicodeUTF8));
        radioButtonGaussian2D5->setText(QApplication::translate("QVSControlPanel", "5", 0, QApplication::UnicodeUTF8));
        radioButtonGaussian2D7->setText(QApplication::translate("QVSControlPanel", "7", 0, QApplication::UnicodeUTF8));
        groupBoxVisibility->setTitle(QApplication::translate("QVSControlPanel", "6. Visibility", 0, QApplication::UnicodeUTF8));
        labelViewEntropy->setText(QApplication::translate("QVSControlPanel", "View Entropy", 0, QApplication::UnicodeUTF8));
        labelViewEntropyValue->setText(QApplication::translate("QVSControlPanel", "0.0", 0, QApplication::UnicodeUTF8));
        labelNorthernHemisphere->setText(QApplication::translate("QVSControlPanel", "N", 0, QApplication::UnicodeUTF8));
        labelSouthernHemisphere->setText(QApplication::translate("QVSControlPanel", "S", 0, QApplication::UnicodeUTF8));
        groupBoxEditor->setTitle(QApplication::translate("QVSControlPanel", "5. Transfer Function Editor", 0, QApplication::UnicodeUTF8));
    } // retranslateUi

};

namespace Ui {
    class QVSControlPanel: public Ui_QVSControlPanel {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_QVSCONTROLPANEL_H
