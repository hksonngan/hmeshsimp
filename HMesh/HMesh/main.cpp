#include "hmesh.h"
#include <QtGui/QApplication>
#include "utils_for_r.h"

int main(int argc, char *argv[])
{
	//comparePlyTris("d:/bun_zipper.ply", "d:/bun_zipper.tris");

	QApplication a(argc, argv);
	
	HMesh w;
	w.show();
	return a.exec();
}
