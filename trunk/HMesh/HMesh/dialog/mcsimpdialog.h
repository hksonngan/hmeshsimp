#ifndef QMCSIMPDIALOG_H
#define QMCSIMPDIALOG_H

#include <QtGui/QDialog>
#include "ui_mcsimpdialog.h"

class QLabel;
class QPushButton;

class QMCSimpDialog : public QDialog, public Ui::MCSimpDialog {
private:
	//QLabel *fileLabel, *isoValueLabel, *decimateRateLabel;
	//QPushButton *openFileButton, *decimateButton;
public:
	QMCSimpDialog();
	~QMCSimpDialog();
};

#endif