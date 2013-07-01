/*
 *  Qt Dialog for Marching Cubes
 *
 *  Author: Ht
 *  Email : waytofall916 at gmail dot com
 *
 *  Copyright (C) Ht. All rights reserved.
 */

#ifndef QMCDialog_H
#define QMCDialog_H

#include <string>
#include <QtGui/QDialog>
#include "ui_mcdialog.h"
#include "vol_set.h"

class QString;
using std::string;

// Qt Dialog for Marching Cubes
class QMCDialog : public QDialog, public Ui::MCDialog 
{
	Q_OBJECT

public:
	QMCDialog(QWidget *parent = NULL);
	~QMCDialog();

private:
	void validateInput();

signals:
	void mcParams(string filename, double isovalue, int* sampleStride);	

public slots:
	void onAccept();
	void onOpenFile();
	void onIsoValueChanged(int value);

private:
    QRegExpValidator *floatValidator;
    QRegExpValidator *uintValidator;
	static const int sliderMax = 2000;
	QString fileName;
	QDataFormat dataFormat;
	double isoValue;
    int sampleStride[3];
};

#endif