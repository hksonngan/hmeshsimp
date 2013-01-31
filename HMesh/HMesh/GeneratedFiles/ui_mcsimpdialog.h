/********************************************************************************
** Form generated from reading UI file 'mcsimpdialog.ui'
**
** Created: Mon Jan 28 22:16:46 2013
**      by: Qt User Interface Compiler version 4.7.2
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_MCSIMPDIALOG_H
#define UI_MCSIMPDIALOG_H

#include <QtCore/QVariant>
#include <QtGui/QAction>
#include <QtGui/QApplication>
#include <QtGui/QButtonGroup>
#include <QtGui/QComboBox>
#include <QtGui/QDialog>
#include <QtGui/QDialogButtonBox>
#include <QtGui/QHeaderView>
#include <QtGui/QLabel>
#include <QtGui/QPushButton>
#include <QtGui/QSlider>

QT_BEGIN_NAMESPACE

class Ui_MCSimpDialog
{
public:
    QDialogButtonBox *buttonBox;
    QPushButton *fileButton;
    QLabel *fileLabel;
    QLabel *isoValueLabel;
    QLabel *decimateRateLabel;
    QSlider *isoValueSlider;
    QSlider *decimateRateSlider;
    QLabel *rateNumLabel;
    QLabel *bufSizeLabel;
    QComboBox *bufSizeComboBox;
    QLabel *isoValueDispLabel;

    void setupUi(QDialog *MCSimpDialog)
    {
        if (MCSimpDialog->objectName().isEmpty())
            MCSimpDialog->setObjectName(QString::fromUtf8("MCSimpDialog"));
        MCSimpDialog->resize(405, 265);
        QSizePolicy sizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);
        sizePolicy.setHorizontalStretch(0);
        sizePolicy.setVerticalStretch(0);
        sizePolicy.setHeightForWidth(MCSimpDialog->sizePolicy().hasHeightForWidth());
        MCSimpDialog->setSizePolicy(sizePolicy);
        buttonBox = new QDialogButtonBox(MCSimpDialog);
        buttonBox->setObjectName(QString::fromUtf8("buttonBox"));
        buttonBox->setGeometry(QRect(30, 210, 341, 32));
        buttonBox->setOrientation(Qt::Horizontal);
        buttonBox->setStandardButtons(QDialogButtonBox::Cancel|QDialogButtonBox::Ok);
        fileButton = new QPushButton(MCSimpDialog);
        fileButton->setObjectName(QString::fromUtf8("fileButton"));
        fileButton->setGeometry(QRect(30, 30, 75, 23));
        fileLabel = new QLabel(MCSimpDialog);
        fileLabel->setObjectName(QString::fromUtf8("fileLabel"));
        fileLabel->setGeometry(QRect(150, 20, 221, 41));
        fileLabel->setWordWrap(true);
        isoValueLabel = new QLabel(MCSimpDialog);
        isoValueLabel->setObjectName(QString::fromUtf8("isoValueLabel"));
        isoValueLabel->setGeometry(QRect(30, 80, 61, 16));
        decimateRateLabel = new QLabel(MCSimpDialog);
        decimateRateLabel->setObjectName(QString::fromUtf8("decimateRateLabel"));
        decimateRateLabel->setGeometry(QRect(30, 120, 91, 16));
        isoValueSlider = new QSlider(MCSimpDialog);
        isoValueSlider->setObjectName(QString::fromUtf8("isoValueSlider"));
        isoValueSlider->setGeometry(QRect(150, 80, 160, 19));
        isoValueSlider->setSingleStep(1);
        isoValueSlider->setValue(50);
        isoValueSlider->setOrientation(Qt::Horizontal);
        decimateRateSlider = new QSlider(MCSimpDialog);
        decimateRateSlider->setObjectName(QString::fromUtf8("decimateRateSlider"));
        decimateRateSlider->setGeometry(QRect(150, 120, 160, 19));
        decimateRateSlider->setSingleStep(1);
        decimateRateSlider->setValue(50);
        decimateRateSlider->setOrientation(Qt::Horizontal);
        rateNumLabel = new QLabel(MCSimpDialog);
        rateNumLabel->setObjectName(QString::fromUtf8("rateNumLabel"));
        rateNumLabel->setGeometry(QRect(320, 120, 61, 16));
        bufSizeLabel = new QLabel(MCSimpDialog);
        bufSizeLabel->setObjectName(QString::fromUtf8("bufSizeLabel"));
        bufSizeLabel->setGeometry(QRect(30, 160, 81, 16));
        bufSizeComboBox = new QComboBox(MCSimpDialog);
        bufSizeComboBox->setObjectName(QString::fromUtf8("bufSizeComboBox"));
        bufSizeComboBox->setGeometry(QRect(150, 160, 101, 21));
        bufSizeComboBox->setEditable(true);
        isoValueDispLabel = new QLabel(MCSimpDialog);
        isoValueDispLabel->setObjectName(QString::fromUtf8("isoValueDispLabel"));
        isoValueDispLabel->setGeometry(QRect(320, 80, 61, 16));

        retranslateUi(MCSimpDialog);
        QObject::connect(buttonBox, SIGNAL(accepted()), MCSimpDialog, SLOT(accept()));
        QObject::connect(buttonBox, SIGNAL(rejected()), MCSimpDialog, SLOT(reject()));

        QMetaObject::connectSlotsByName(MCSimpDialog);
    } // setupUi

    void retranslateUi(QDialog *MCSimpDialog)
    {
        MCSimpDialog->setWindowTitle(QApplication::translate("MCSimpDialog", "Params", 0, QApplication::UnicodeUTF8));
        fileButton->setText(QApplication::translate("MCSimpDialog", "Open", 0, QApplication::UnicodeUTF8));
        fileLabel->setText(QApplication::translate("MCSimpDialog", "Volume File ...", 0, QApplication::UnicodeUTF8));
        isoValueLabel->setText(QApplication::translate("MCSimpDialog", "Iso Value", 0, QApplication::UnicodeUTF8));
        decimateRateLabel->setText(QApplication::translate("MCSimpDialog", "Decimate Rate", 0, QApplication::UnicodeUTF8));
        rateNumLabel->setText(QApplication::translate("MCSimpDialog", "50%", 0, QApplication::UnicodeUTF8));
        bufSizeLabel->setText(QApplication::translate("MCSimpDialog", "Buffer Size", 0, QApplication::UnicodeUTF8));
        isoValueDispLabel->setText(QApplication::translate("MCSimpDialog", "--", 0, QApplication::UnicodeUTF8));
    } // retranslateUi

};

namespace Ui {
    class MCSimpDialog: public Ui_MCSimpDialog {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_MCSIMPDIALOG_H
