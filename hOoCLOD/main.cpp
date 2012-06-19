#include "hooclod.h"
#include <QtGui/QApplication>

#include "spatial_division.h"

int main(int argc, char *argv[])
{
	HSpatialDivision sd;

	QApplication a(argc, argv);
	hOoCLOD w;
	w.show();
	return a.exec();
}
