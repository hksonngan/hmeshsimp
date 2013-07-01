#include "mcsimpdialog.h"
#include <QtGui/QLabel>
#include <QtGui/QPushButton>
#include <QtGui/QFileDialog>
#include <QtCore/QRegExp>
#include <QtGui/QRegExpValidator>
#include <QtGui/QMessageBox>
#include "limits.h"

QMCSimpDialog::QMCSimpDialog(QWidget *parent):
	QDialog(parent),
	dataFormat(DATA_UNKNOWN) {
	setupUi(this);
	buttonBox->button(QDialogButtonBox::Ok)->setEnabled(false);
	//setFixedHeight(sizeHint().height());

	for (int i = 1000; i <= 20000; i += 1000) {
		bufSizeComboBox->addItem(QString::number(i, 10));
	}

	isoValueSlider->setMaximum(sliderMax);
	isoValueSlider->setValue(sliderMax/2);
	decimateRateSlider->setMaximum(sliderMax);
	decimateRateSlider->setValue(sliderMax/2);
	onDecimateRateChanged(sliderMax/2);
	
	numValidator = new QRegExpValidator(QRegExp("^\\+?0*[1-9]\\d*$"), this);
	bufSizeComboBox->setValidator(numValidator);
    floatValidator = new QRegExpValidator(QRegExp("^[+-]?(?:0*[1-9]\\d*.\\d*|\\d*.0*[1-9]\\d*|0*[1-9]\\d*)$"), this);
    isoValueLineEdit->setValidator(floatValidator);
    decimateRateLineEdit->setValidator(floatValidator);
    uintValidator = new QRegExpValidator(QRegExp("^[1-9]\\d*$"), this);
    sampleStrideXLineEdit->setValidator(uintValidator);
    sampleStrideYLineEdit->setValidator(uintValidator);
    sampleStrideZLineEdit->setValidator(uintValidator);

	connect(buttonBox, SIGNAL(accepted()), this, SLOT(onAccept()));
	connect(fileButton, SIGNAL(clicked()), this, SLOT(onOpenFile()));
	connect(isoValueSlider, SIGNAL(valueChanged(int)), this, SLOT(onIsoValueChanged(int)));
	connect(decimateRateSlider, SIGNAL(valueChanged(int)), this, SLOT(onDecimateRateChanged(int)));
	connect(bufSizeComboBox, SIGNAL(editTextChanged(const QString &)), this, SLOT(onBufSizeChanged(const QString &)));
}

QMCSimpDialog::~QMCSimpDialog() {
	
}

void QMCSimpDialog::onAccept() {
    sampleStride[0] = sampleStrideXLineEdit->text().toInt();
    sampleStride[1] = sampleStrideYLineEdit->text().toInt();
    sampleStride[2] = sampleStrideZLineEdit->text().toInt();
    isoValue = isoValueLineEdit->text().toDouble();
    decimateRate = decimateRateLineEdit->text().toDouble() / 100.0;
	emit mcsimpParams(
			string(fileName.toLocal8Bit().data()), isoValue, sampleStride, 
            decimateRate, bufSizeComboBox->currentText().toInt());
}

void QMCSimpDialog::onOpenFile() {
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

void QMCSimpDialog::onIsoValueChanged(int value) {
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

void QMCSimpDialog::onDecimateRateChanged(int value) {
	double _decimateRate = (double)value / (double)sliderMax;
	decimateRateLineEdit->setText(QString::number(_decimateRate * 100, 'f', 2));
}

void QMCSimpDialog::onBufSizeChanged(const QString & text) {
	validateInput();
}

void QMCSimpDialog::validateInput() {
	int pos = 0;
	if (numValidator->validate(bufSizeComboBox->currentText(), pos) == QValidator::Acceptable && fileName != QString())
		buttonBox->button(QDialogButtonBox::Ok)->setEnabled(true);
	else
		buttonBox->button(QDialogButtonBox::Ok)->setEnabled(false);
}