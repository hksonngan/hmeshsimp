/********************************************************************************
** Form generated from reading UI file 'hmesh.ui'
**
** Created: Fri Jul 27 15:34:25 2012
**      by: Qt User Interface Compiler version 4.7.2
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_HMESH_H
#define UI_HMESH_H

#include <QtCore/QVariant>
#include <QtGui/QAction>
#include <QtGui/QApplication>
#include <QtGui/QButtonGroup>
#include <QtGui/QHeaderView>
#include <QtGui/QMainWindow>
#include <QtGui/QMenuBar>
#include <QtGui/QStatusBar>
#include <QtGui/QToolBar>
#include <QtGui/QWidget>

QT_BEGIN_NAMESPACE

class Ui_HMeshClass
{
public:
    QMenuBar *menuBar;
    QToolBar *mainToolBar;
    QWidget *centralWidget;
    QStatusBar *statusBar;

    void setupUi(QMainWindow *HMeshClass)
    {
        if (HMeshClass->objectName().isEmpty())
            HMeshClass->setObjectName(QString::fromUtf8("HMeshClass"));
        HMeshClass->resize(600, 400);
        menuBar = new QMenuBar(HMeshClass);
        menuBar->setObjectName(QString::fromUtf8("menuBar"));
        HMeshClass->setMenuBar(menuBar);
        mainToolBar = new QToolBar(HMeshClass);
        mainToolBar->setObjectName(QString::fromUtf8("mainToolBar"));
        HMeshClass->addToolBar(mainToolBar);
        centralWidget = new QWidget(HMeshClass);
        centralWidget->setObjectName(QString::fromUtf8("centralWidget"));
        HMeshClass->setCentralWidget(centralWidget);
        statusBar = new QStatusBar(HMeshClass);
        statusBar->setObjectName(QString::fromUtf8("statusBar"));
        HMeshClass->setStatusBar(statusBar);

        retranslateUi(HMeshClass);

        QMetaObject::connectSlotsByName(HMeshClass);
    } // setupUi

    void retranslateUi(QMainWindow *HMeshClass)
    {
        HMeshClass->setWindowTitle(QApplication::translate("HMeshClass", "HMesh", 0, QApplication::UnicodeUTF8));
    } // retranslateUi

};

namespace Ui {
    class HMeshClass: public Ui_HMeshClass {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_HMESH_H
