/********************************************************************************
** Form generated from reading UI file 'QVisualizer.ui'
**
** Created: Wed Jan 2 15:03:25 2013
**      by: Qt User Interface Compiler version 4.7.2
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_QVISUALIZER_H
#define UI_QVISUALIZER_H

#include <QtCore/QVariant>
#include <QtGui/QAction>
#include <QtGui/QApplication>
#include <QtGui/QButtonGroup>
#include <QtGui/QDockWidget>
#include <QtGui/QHeaderView>
#include <QtGui/QMainWindow>
#include <QtGui/QMenu>
#include <QtGui/QMenuBar>
#include <QtGui/QStatusBar>
#include <QtGui/QWidget>

QT_BEGIN_NAMESPACE

class Ui_QVisualizerClass
{
public:
    QAction *actionVector_Field_Visualization;
    QAction *actionMarching_Cubes;
    QAction *actionVolume_Rendering;
    QAction *actionTensor_Field_Visualization;
    QAction *actionVirtual_Environment_For_Visualization;
    QAction *actionLarge_Scale_Data_Visualization;
    QAction *actionVisualization_Software_and_Frameworks;
    QAction *actionPerceptual_Issues_in_Visualization;
    QAction *actionSelected_Topics_and_QApplications;
    QWidget *centralwidget;
    QMenuBar *menubar;
    QMenu *menuFile;
    QMenu *menuEdit;
    QMenu *menuView;
    QMenu *menuSettings;
    QMenu *menuTopics;
    QMenu *menuScalar_Field_Visualization;
    QMenu *menuAbout;
    QStatusBar *statusbar;
    QDockWidget *dockWidget;
    QWidget *dockWidgetContents;

    void setupUi(QMainWindow *QVisualizerClass)
    {
        if (QVisualizerClass->objectName().isEmpty())
            QVisualizerClass->setObjectName(QString::fromUtf8("QVisualizerClass"));
        QVisualizerClass->resize(512, 512);
        QVisualizerClass->setMaximumSize(QSize(16777215, 16777215));
        actionVector_Field_Visualization = new QAction(QVisualizerClass);
        actionVector_Field_Visualization->setObjectName(QString::fromUtf8("actionVector_Field_Visualization"));
        actionMarching_Cubes = new QAction(QVisualizerClass);
        actionMarching_Cubes->setObjectName(QString::fromUtf8("actionMarching_Cubes"));
        actionVolume_Rendering = new QAction(QVisualizerClass);
        actionVolume_Rendering->setObjectName(QString::fromUtf8("actionVolume_Rendering"));
        actionTensor_Field_Visualization = new QAction(QVisualizerClass);
        actionTensor_Field_Visualization->setObjectName(QString::fromUtf8("actionTensor_Field_Visualization"));
        actionVirtual_Environment_For_Visualization = new QAction(QVisualizerClass);
        actionVirtual_Environment_For_Visualization->setObjectName(QString::fromUtf8("actionVirtual_Environment_For_Visualization"));
        actionLarge_Scale_Data_Visualization = new QAction(QVisualizerClass);
        actionLarge_Scale_Data_Visualization->setObjectName(QString::fromUtf8("actionLarge_Scale_Data_Visualization"));
        actionVisualization_Software_and_Frameworks = new QAction(QVisualizerClass);
        actionVisualization_Software_and_Frameworks->setObjectName(QString::fromUtf8("actionVisualization_Software_and_Frameworks"));
        actionPerceptual_Issues_in_Visualization = new QAction(QVisualizerClass);
        actionPerceptual_Issues_in_Visualization->setObjectName(QString::fromUtf8("actionPerceptual_Issues_in_Visualization"));
        actionSelected_Topics_and_QApplications = new QAction(QVisualizerClass);
        actionSelected_Topics_and_QApplications->setObjectName(QString::fromUtf8("actionSelected_Topics_and_QApplications"));
        centralwidget = new QWidget(QVisualizerClass);
        centralwidget->setObjectName(QString::fromUtf8("centralwidget"));
        QVisualizerClass->setCentralWidget(centralwidget);
        menubar = new QMenuBar(QVisualizerClass);
        menubar->setObjectName(QString::fromUtf8("menubar"));
        menubar->setGeometry(QRect(0, 0, 512, 23));
        menuFile = new QMenu(menubar);
        menuFile->setObjectName(QString::fromUtf8("menuFile"));
        menuEdit = new QMenu(menubar);
        menuEdit->setObjectName(QString::fromUtf8("menuEdit"));
        menuView = new QMenu(menubar);
        menuView->setObjectName(QString::fromUtf8("menuView"));
        menuSettings = new QMenu(menubar);
        menuSettings->setObjectName(QString::fromUtf8("menuSettings"));
        menuTopics = new QMenu(menubar);
        menuTopics->setObjectName(QString::fromUtf8("menuTopics"));
        menuScalar_Field_Visualization = new QMenu(menuTopics);
        menuScalar_Field_Visualization->setObjectName(QString::fromUtf8("menuScalar_Field_Visualization"));
        menuAbout = new QMenu(menubar);
        menuAbout->setObjectName(QString::fromUtf8("menuAbout"));
        QVisualizerClass->setMenuBar(menubar);
        statusbar = new QStatusBar(QVisualizerClass);
        statusbar->setObjectName(QString::fromUtf8("statusbar"));
        QVisualizerClass->setStatusBar(statusbar);
        dockWidget = new QDockWidget(QVisualizerClass);
        dockWidget->setObjectName(QString::fromUtf8("dockWidget"));
        dockWidget->setMinimumSize(QSize(60, 41));
        dockWidget->setFloating(true);
        dockWidget->setFeatures(QDockWidget::DockWidgetFloatable|QDockWidget::DockWidgetMovable);
        dockWidget->setAllowedAreas(Qt::BottomDockWidgetArea);
        dockWidgetContents = new QWidget();
        dockWidgetContents->setObjectName(QString::fromUtf8("dockWidgetContents"));
        dockWidget->setWidget(dockWidgetContents);
        QVisualizerClass->addDockWidget(static_cast<Qt::DockWidgetArea>(8), dockWidget);

        menubar->addAction(menuFile->menuAction());
        menubar->addAction(menuEdit->menuAction());
        menubar->addAction(menuView->menuAction());
        menubar->addAction(menuSettings->menuAction());
        menubar->addAction(menuTopics->menuAction());
        menubar->addAction(menuAbout->menuAction());
        menuTopics->addAction(menuScalar_Field_Visualization->menuAction());
        menuTopics->addAction(actionVector_Field_Visualization);
        menuTopics->addAction(actionTensor_Field_Visualization);
        menuTopics->addAction(actionVirtual_Environment_For_Visualization);
        menuTopics->addAction(actionLarge_Scale_Data_Visualization);
        menuTopics->addAction(actionVisualization_Software_and_Frameworks);
        menuTopics->addAction(actionPerceptual_Issues_in_Visualization);
        menuTopics->addAction(actionSelected_Topics_and_QApplications);
        menuScalar_Field_Visualization->addAction(actionMarching_Cubes);
        menuScalar_Field_Visualization->addAction(actionVolume_Rendering);

        retranslateUi(QVisualizerClass);

        QMetaObject::connectSlotsByName(QVisualizerClass);
    } // setupUi

    void retranslateUi(QMainWindow *QVisualizerClass)
    {
        QVisualizerClass->setWindowTitle(QApplication::translate("QVisualizerClass", "QVisualizer", 0, QApplication::UnicodeUTF8));
        actionVector_Field_Visualization->setText(QApplication::translate("QVisualizerClass", "Vector Field Visualization", 0, QApplication::UnicodeUTF8));
        actionMarching_Cubes->setText(QApplication::translate("QVisualizerClass", "Marching Cubes", 0, QApplication::UnicodeUTF8));
        actionVolume_Rendering->setText(QApplication::translate("QVisualizerClass", "Volume Rendering", 0, QApplication::UnicodeUTF8));
        actionTensor_Field_Visualization->setText(QApplication::translate("QVisualizerClass", "Tensor Field Visualization", 0, QApplication::UnicodeUTF8));
        actionVirtual_Environment_For_Visualization->setText(QApplication::translate("QVisualizerClass", "Virtual Environment For Visualization", 0, QApplication::UnicodeUTF8));
        actionLarge_Scale_Data_Visualization->setText(QApplication::translate("QVisualizerClass", "Large-Scale Data Visualization", 0, QApplication::UnicodeUTF8));
        actionVisualization_Software_and_Frameworks->setText(QApplication::translate("QVisualizerClass", "Visualization Software and Frameworks", 0, QApplication::UnicodeUTF8));
        actionPerceptual_Issues_in_Visualization->setText(QApplication::translate("QVisualizerClass", "Perceptual Issues in Visualization", 0, QApplication::UnicodeUTF8));
        actionSelected_Topics_and_QApplications->setText(QApplication::translate("QVisualizerClass", "Selected Topics and QApplications", 0, QApplication::UnicodeUTF8));
        menuFile->setTitle(QApplication::translate("QVisualizerClass", "File", 0, QApplication::UnicodeUTF8));
        menuEdit->setTitle(QApplication::translate("QVisualizerClass", "Edit", 0, QApplication::UnicodeUTF8));
        menuView->setTitle(QApplication::translate("QVisualizerClass", "View", 0, QApplication::UnicodeUTF8));
        menuSettings->setTitle(QApplication::translate("QVisualizerClass", "Settings", 0, QApplication::UnicodeUTF8));
        menuTopics->setTitle(QApplication::translate("QVisualizerClass", "Topics", 0, QApplication::UnicodeUTF8));
        menuScalar_Field_Visualization->setTitle(QApplication::translate("QVisualizerClass", "Scalar Field Visualization", 0, QApplication::UnicodeUTF8));
        menuAbout->setTitle(QApplication::translate("QVisualizerClass", "About", 0, QApplication::UnicodeUTF8));
        dockWidget->setWindowTitle(QString());
    } // retranslateUi

};

namespace Ui {
    class QVisualizerClass: public Ui_QVisualizerClass {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_QVISUALIZER_H
