#include "mcdialog.h"
#include <QtGui/QLabel>
#include <QtGui/QPushButton>
#include <QtGui/QFileDialog>
#include <QtGui/QMessageBox>
#include "limits.h"

QMCDialog::QMCDialog(QWidget *parent):
	QDialog(parent),
	dataFormat(DATA_UNKNOWN) {
	setupUi(this);
	buttonBox->button(QDialogButtonBox::Ok)->setEnabled(false);
	//setFixedHeight(sizeHint().height());

	isoValueSlider->setMaximum(sliderMax);
	isoValueSlider->setValue(sliderMax/2);

    floatValidator = new QRegExpValidator(QRegExp("^[+-]?(?:0*[1-9]\\d*.\\d*|\\d*.0*[1-9]\\d*|0*[1-9]\\d*)$"), this);
    isoValueLineEdit->setValidator(floatValidator);
    uintValidator = new QRegExpValidator(QRegExp("^[1-9]\\d*$"), this);
    sampleStrideXLineEdit->setValidator(uintValidator);
    sampleStrideYLineEdit->setValidator(uintValidator);
    sampleStrideZLineEdit->setValidator(uintValidator);
	
	connect(buttonBox, SIGNAL(accepted()), this, SLOT(onAccept()));
	connect(fileButton, SIGNAL(clicked()), this, SLOT(onOpenFile()));
	connect(isoValueSlider, SIGNAL(valueChanged(int)), this, SLOT(onIsoValueChanged(int)));
    validateInput();
}

QMCDialog::~QMCDialog() {
	
}

void QMCDialog::onAccept() {
    sampleStride[0] = sampleStrideXLineEdit->text().toInt();
    sampleStride[1] = sampleStrideYLineEdit->text().toInt();
    sampleStride[2] = sampleStrideZLineEdit->text().toInt();
    isoValue = isoValueLineEdit->text().toDouble();
	emit mcParams(string(fileName.toLocal8Bit().data()), isoValue, sampleStride);
}

void QMCDialog::onOpenFile() {
	QString retName = QFileDialog::getOpenFileName(this,
		tr("Open Volume File"), fileName,
		tr("Volume Files(*.dat);;"
		"All Files(*.*)"));
	if (retName == QString())
		return;

	VolumeSet volSet;
	if (!volSet.parseDataFile(string(retName.toLocal8Bit().data()), false)) {
		QMessageBox::warning(this, tr("Warning"),
			tr("Unable to parse .dat file"),
			QMessageBox::Ok);
		return;
	}
	
	fileName = retName;
	fileLabel->setText(fileName);
    resValLabel->setText(
        QString::number(volSet.volumeSize.s[0]) + " x " + 
        QString::number(volSet.volumeSize.s[1]) + " x " + 
        QString::number(volSet.volumeSize.s[2]));
	dataFormat = volSet.format;
	onIsoValueChanged(isoValueSlider->value());
	validateInput();
}

void QMCDialog::onIsoValueChanged(int value) {
    float _isoValue;
    switch(dataFormat) {
        case DATA_CHAR:
            _isoValue = ((double)(CHAR_MAX - CHAR_MIN) * ((double)value / (double)sliderMax)) + CHAR_MIN;
            isoValueLineEdit->setText(QString::number(_isoValue, 'f', 2));
            break;
        case DATA_UCHAR:
            _isoValue = ((double)UCHAR_MAX * ((double)value / (double)sliderMax));
            isoValueLineEdit->setText(QString::number(_isoValue, 'f', 2));
            break;
        case DATA_SHORT:
            _isoValue = ((double)(SHRT_MAX - SHRT_MIN) * ((double)value / (double)sliderMax)) + SHRT_MIN;
            isoValueLineEdit->setText(QString::number(_isoValue, 'f', 2));
            break;
        case DATA_USHORT:
            _isoValue = ((double)USHRT_MAX * ((double)value / (double)sliderMax));
            isoValueLineEdit->setText(QString::number(_isoValue, 'f', 2));
            break;
    }
}

void QMCDialog::validateInput() {
	int pos = 0;
	if (fileName != QString())
		buttonBox->button(QDialogButtonBox::Ok)->setEnabled(true);
	else
		buttonBox->button(QDialogButtonBox::Ok)->setEnabled(false);
}