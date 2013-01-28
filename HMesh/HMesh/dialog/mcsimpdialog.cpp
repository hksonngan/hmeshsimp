#include "mcsimpdialog.h"
#include <QtGui/QLabel>
#include <QtGui/QPushButton>
#include <QtGui/QVBoxLayout>
#include <QtGui/QHBoxLayout>
#include <QtGui/QDialogButtonBox>
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

	connect(buttonBox, SIGNAL(accepted()), this, SLOT(onAccept()));
	connect(fileButton, SIGNAL(clicked()), this, SLOT(onOpenFile()));
	connect(isoValueSlider, SIGNAL(valueChanged(int)), this, SLOT(onIsoValueChanged(int)));
	connect(decimateRateSlider, SIGNAL(valueChanged(int)), this, SLOT(onDecimateRateChanged(int)));
	connect(bufSizeComboBox, SIGNAL(editTextChanged(const QString &)), this, SLOT(onBufSizeChanged(const QString &)));
}

QMCSimpDialog::~QMCSimpDialog() {
	
}

void QMCSimpDialog::onAccept() {
	emit mcsimpParams(string(fileName.toLocal8Bit().data()), isoValue, decimateRate, bufSizeComboBox->currentText().toInt());
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
	dataFormat = volSet.format;
	onIsoValueChanged(isoValueSlider->value());
	validateInput();
}

void QMCSimpDialog::onIsoValueChanged(int value) {
	switch(dataFormat) {
		case DATA_UCHAR:
			isoValue = ((double)UCHAR_MAX * ((double)value / (double)sliderMax));
			isoValueDispLabel->setText(QString::number(isoValue, 'f', 2));
			break;
		case DATA_USHORT:
			isoValue = ((double)USHRT_MAX * ((double)value / (double)sliderMax));
			isoValueDispLabel->setText(QString::number(isoValue, 'f', 2));
			break;
	}
}

void QMCSimpDialog::onDecimateRateChanged(int value) {
	decimateRate = (double)value / (double)sliderMax;
	rateNumLabel->setText(QString::number(decimateRate * 100, 'f', 2) + "%");
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