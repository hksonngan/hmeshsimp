/*
 *  Qt Dialog for On-the-fly Simplification of Marching Cubes Iso-surfaces
 *
 *  Author: Ht
 *  Email : waytofall916 at gmail dot com
 *
 *  Copyright (C) Ht. All rights reserved.
 */

#ifndef QMCSIMPDIALOG_H
#define QMCSIMPDIALOG_H

#include <string>
#include <QtGui/QDialog>
#include "ui_mcsimpdialog.h"
#include "vol_set.h"

class QString;
using std::string;

// Qt Dialog for On-the-fly Simplification of Marching Cubes Iso-surfaces
class QMCSimpDialog : public QDialog, public Ui::MCSimpDialog 
{
	Q_OBJECT

public:
	QMCSimpDialog(QWidget *parent = NULL);
	~QMCSimpDialog();

private:
	void validateInput();

signals:
	void mcsimpParams(string filename, double isovalue, 
        int* sampleStride, double deimateRate, unsigned int maxNewTri);	

public slots:
	void onAccept();
	void onOpenFile();
	void onIsoValueChanged(int value);
	void onDecimateRateChanged(int value);
	void onBufSizeChanged(const QString & text);

private:
	static const int sliderMax = 2000;
	QString fileName;
	QDataFormat dataFormat;
	double isoValue;
	double decimateRate;
    int sampleStride[3];
	QRegExpValidator *numValidator;
    QRegExpValidator *floatValidator;
    QRegExpValidator *uintValidator;
};

#endif