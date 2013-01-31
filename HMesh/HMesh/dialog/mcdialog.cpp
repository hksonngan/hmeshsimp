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
	
	connect(buttonBox, SIGNAL(accepted()), this, SLOT(onAccept()));
	connect(fileButton, SIGNAL(clicked()), this, SLOT(onOpenFile()));
	connect(isoValueSlider, SIGNAL(valueChanged(int)), this, SLOT(onIsoValueChanged(int)));
}

QMCDialog::~QMCDialog() {
	
}

void QMCDialog::onAccept() {
	emit mcParams(string(fileName.toLocal8Bit().data()), isoValue);
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
	dataFormat = volSet.format;
	onIsoValueChanged(isoValueSlider->value());
	validateInput();
}

void QMCDialog::onIsoValueChanged(int value) {
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

void QMCDialog::validateInput() {
	int pos = 0;
	if (fileName != QString())
		buttonBox->button(QDialogButtonBox::Ok)->setEnabled(true);
	else
		buttonBox->button(QDialogButtonBox::Ok)->setEnabled(false);
}