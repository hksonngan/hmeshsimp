/********************************************************************************
** Form generated from reading UI file 'mcdialog.ui'
**
** Created: Wed Jan 30 19:43:22 2013
**      by: Qt User Interface Compiler version 4.7.2
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_MCDIALOG_H
#define UI_MCDIALOG_H

#include <QtCore/QVariant>
#include <QtGui/QAction>
#include <QtGui/QApplication>
#include <QtGui/QButtonGroup>
#include <QtGui/QDialog>
#include <QtGui/QDialogButtonBox>
#include <QtGui/QHeaderView>
#include <QtGui/QLabel>
#include <QtGui/QPushButton>
#include <QtGui/QSlider>

QT_BEGIN_NAMESPACE

class Ui_MCDialog
{
public:
    QDialogButtonBox *buttonBox;
    QPushButton *fileButton;
    QLabel *fileLabel;
    QLabel *isoValueLabel;
    QSlider *isoValueSlider;
    QLabel *isoValueDispLabel;

    void setupUi(QDialog *MCDialog)
    {
        if (MCDialog->objectName().isEmpty())
            MCDialog->setObjectName(QString::fromUtf8("MCDialog"));
        MCDialog->resize(405, 164);
        QSizePolicy sizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);
        sizePolicy.setHorizontalStretch(0);
        sizePolicy.setVerticalStretch(0);
        sizePolicy.setHeightForWidth(MCDialog->sizePolicy().hasHeightForWidth());
        MCDialog->setSizePolicy(sizePolicy);
        buttonBox = new QDialogButtonBox(MCDialog);
        buttonBox->setObjectName(QString::fromUtf8("buttonBox"));
        buttonBox->setGeometry(QRect(30, 120, 341, 32));
        buttonBox->setOrientation(Qt::Horizontal);
        buttonBox->setStandardButtons(QDialogButtonBox::Cancel|QDialogButtonBox::Ok);
        fileButton = new QPushButton(MCDialog);
        fileButton->setObjectName(QString::fromUtf8("fileButton"));
        fileButton->setGeometry(QRect(30, 30, 75, 23));
        fileLabel = new QLabel(MCDialog);
        fileLabel->setObjectName(QString::fromUtf8("fileLabel"));
        fileLabel->setGeometry(QRect(150, 20, 221, 41));
        fileLabel->setWordWrap(true);
        isoValueLabel = new QLabel(MCDialog);
        isoValueLabel->setObjectName(QString::fromUtf8("isoValueLabel"));
        isoValueLabel->setGeometry(QRect(30, 80, 61, 16));
        isoValueSlider = new QSlider(MCDialog);
        isoValueSlider->setObjectName(QString::fromUtf8("isoValueSlider"));
        isoValueSlider->setGeometry(QRect(150, 80, 160, 19));
        isoValueSlider->setSingleStep(1);
        isoValueSlider->setValue(50);
        isoValueSlider->setOrientation(Qt::Horizontal);
        isoValueDispLabel = new QLabel(MCDialog);
        isoValueDispLabel->setObjectName(QString::fromUtf8("isoValueDispLabel"));
        isoValueDispLabel->setGeometry(QRect(320, 80, 61, 16));

        retranslateUi(MCDialog);
        QObject::connect(buttonBox, SIGNAL(accepted()), MCDialog, SLOT(accept()));
        QObject::connect(buttonBox, SIGNAL(rejected()), MCDialog, SLOT(reject()));

        QMetaObject::connectSlotsByName(MCDialog);
    } // setupUi

    void retranslateUi(QDialog *MCDialog)
    {
        MCDialog->setWindowTitle(QApplication::translate("MCDialog", "Params", 0, QApplication::UnicodeUTF8));
        fileButton->setText(QApplication::translate("MCDialog", "Open", 0, QApplication::UnicodeUTF8));
        fileLabel->setText(QApplication::translate("MCDialog", "Volume File ...", 0, QApplication::UnicodeUTF8));
        isoValueLabel->setText(QApplication::translate("MCDialog", "Iso Value", 0, QApplication::UnicodeUTF8));
        isoValueDispLabel->setText(QApplication::translate("MCDialog", "--", 0, QApplication::UnicodeUTF8));
    } // retranslateUi

};

namespace Ui {
    class MCDialog: public Ui_MCDialog {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_MCDIALOG_H
