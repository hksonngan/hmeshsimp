/*
 *  Qt Dialog for QSlim (nearly deprecated)
 *
 *  Author: Ht
 *  Email : waytofall916 at gmail dot com
 *
 *  Copyright (C) Ht. All rights reserved.
 */

#ifndef QSLIMDIALOG_H
#define QSLIMDIALOG_H

#include <QtGui/QDialog>

class QCheckBox;
class QLabel;
class QLineEdit;
class QPushButton;

// Qt Dialog for QSlim (nearly deprecated)
class QSlimDialog : public QDialog
{
	Q_OBJECT

public:
	QSlimDialog(QWidget *parent = NULL);
	~QSlimDialog();

signals:
	//void findNext(const QString &str, Qt::CaseSensitivity cs);
	//void findPrevious(const QString &str, Qt::CaseSensitivity cs);
	//private slots:
	//void findClicked();
	//void enableFindButton(const QString &text);

private slots:
	void goClicked();

private:
	QLabel *label;
	QLineEdit *lineEdit;
	//QCheckBox *caseCheckBox;
	//QCheckBox *backwardCheckBox;
	QPushButton *goButton;
	//QPushButton *closeButton;

public:
	QLineEdit* getLineEdit() { return this->lineEdit; };
};

#endif // QSLIMDIALOG_H