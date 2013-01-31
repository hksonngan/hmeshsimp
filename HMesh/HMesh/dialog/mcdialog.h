#ifndef QMCDialog_H
#define QMCDialog_H

#include <string>
#include <QtGui/QDialog>
#include "ui_mcdialog.h"
#include "vol_set.h"

class QString;
using std::string;

class QMCDialog : public QDialog, public Ui::MCDialog 
{
	Q_OBJECT

public:
	QMCDialog(QWidget *parent = NULL);
	~QMCDialog();

private:
	void validateInput();

signals:
	void mcParams(string filename, double isovalue);	

public slots:
	void onAccept();
	void onOpenFile();
	void onIsoValueChanged(int value);

private:
	static const int sliderMax = 2000;
	QString fileName;
	QDataFormat dataFormat;
	double isoValue;
};

#endif