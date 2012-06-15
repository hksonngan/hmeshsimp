#ifndef HOOCLOD_H
#define HOOCLOD_H

#include <QtGui/QMainWindow>
#include "ui_hooclod.h"

class hOoCLOD : public QMainWindow
{
	Q_OBJECT

public:
	hOoCLOD(QWidget *parent = 0, Qt::WFlags flags = 0);
	~hOoCLOD();

private:
	Ui::hOoCLODClass ui;
};

#endif // HOOCLOD_H
