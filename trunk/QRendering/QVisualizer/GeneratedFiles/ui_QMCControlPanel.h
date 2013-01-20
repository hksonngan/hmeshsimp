/********************************************************************************
** Form generated from reading UI file 'QMCControlPanel.ui'
**
** Created: Wed Jan 2 15:03:28 2013
**      by: Qt User Interface Compiler version 4.7.2
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_QMCCONTROLPANEL_H
#define UI_QMCCONTROLPANEL_H

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

QT_BEGIN_NAMESPACE

class Ui_QMCControlPanel
{
public:
    QVBoxLayout *verticalLayout;
    QGroupBox *groupBoxIsoValue;
    QHBoxLayout *horizontalLayoutTimeVaryingData;
    QLabel *labelIsoValueTooltip;
    QSlider *horizontalSliderIsoValue;
    QLabel *labelIsoValueData;

    void setupUi(QWidget *QMCControlPanel)
    {
        if (QMCControlPanel->objectName().isEmpty())
            QMCControlPanel->setObjectName(QString::fromUtf8("QMCControlPanel"));
        QMCControlPanel->resize(512, 100);
        QMCControlPanel->setMinimumSize(QSize(512, 0));
        QMCControlPanel->setMaximumSize(QSize(16777215, 16777215));
        verticalLayout = new QVBoxLayout(QMCControlPanel);
#ifndef Q_OS_MAC
        verticalLayout->setSpacing(6);
#endif
        verticalLayout->setContentsMargins(3, 3, 3, 3);
        verticalLayout->setObjectName(QString::fromUtf8("verticalLayout"));
        groupBoxIsoValue = new QGroupBox(QMCControlPanel);
        groupBoxIsoValue->setObjectName(QString::fromUtf8("groupBoxIsoValue"));
        groupBoxIsoValue->setMinimumSize(QSize(0, 60));
        groupBoxIsoValue->setMaximumSize(QSize(16777215, 60));
        horizontalLayoutTimeVaryingData = new QHBoxLayout(groupBoxIsoValue);
        horizontalLayoutTimeVaryingData->setContentsMargins(6, 6, 6, 6);
        horizontalLayoutTimeVaryingData->setObjectName(QString::fromUtf8("horizontalLayoutTimeVaryingData"));
        labelIsoValueTooltip = new QLabel(groupBoxIsoValue);
        labelIsoValueTooltip->setObjectName(QString::fromUtf8("labelIsoValueTooltip"));
        labelIsoValueTooltip->setMinimumSize(QSize(96, 0));
        labelIsoValueTooltip->setMaximumSize(QSize(96, 16777215));
        labelIsoValueTooltip->setAlignment(Qt::AlignCenter);

        horizontalLayoutTimeVaryingData->addWidget(labelIsoValueTooltip);

        horizontalSliderIsoValue = new QSlider(groupBoxIsoValue);
        horizontalSliderIsoValue->setObjectName(QString::fromUtf8("horizontalSliderIsoValue"));
        horizontalSliderIsoValue->setMinimum(0);
        horizontalSliderIsoValue->setMaximum(200);
        horizontalSliderIsoValue->setPageStep(5);
        horizontalSliderIsoValue->setValue(100);
        horizontalSliderIsoValue->setOrientation(Qt::Horizontal);

        horizontalLayoutTimeVaryingData->addWidget(horizontalSliderIsoValue);

        labelIsoValueData = new QLabel(groupBoxIsoValue);
        labelIsoValueData->setObjectName(QString::fromUtf8("labelIsoValueData"));
        labelIsoValueData->setMinimumSize(QSize(32, 0));
        labelIsoValueData->setMaximumSize(QSize(32, 16777215));
        labelIsoValueData->setAlignment(Qt::AlignRight|Qt::AlignTrailing|Qt::AlignVCenter);

        horizontalLayoutTimeVaryingData->addWidget(labelIsoValueData);


        verticalLayout->addWidget(groupBoxIsoValue);


        retranslateUi(QMCControlPanel);
        QObject::connect(horizontalSliderIsoValue, SIGNAL(valueChanged(int)), labelIsoValueData, SLOT(setNum(int)));

        QMetaObject::connectSlotsByName(QMCControlPanel);
    } // setupUi

    void retranslateUi(QWidget *QMCControlPanel)
    {
        QMCControlPanel->setWindowTitle(QApplication::translate("QMCControlPanel", "Control Panel", 0, QApplication::UnicodeUTF8));
        groupBoxIsoValue->setTitle(QApplication::translate("QMCControlPanel", "1. Marching Cubes", 0, QApplication::UnicodeUTF8));
        labelIsoValueTooltip->setText(QApplication::translate("QMCControlPanel", "Iso-Value", 0, QApplication::UnicodeUTF8));
        labelIsoValueData->setText(QApplication::translate("QMCControlPanel", "100", 0, QApplication::UnicodeUTF8));
    } // retranslateUi

};

namespace Ui {
    class QMCControlPanel: public Ui_QMCControlPanel {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_QMCCONTROLPANEL_H
