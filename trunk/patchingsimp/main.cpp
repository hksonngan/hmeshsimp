#include <iostream>
#include <fstream>

#include "divide_grid_mesh.h"
#include "trivial.h"

using std::ofstream;
using std::fstream;
using std::cout;
using std::endl;

char* filename;
int x_div, y_div, z_div;

int main(int argc, char** argv)
{
	HMeshGridDivide mesh_divide;
	ofstream flog("patchingsimp.log", fstream::app);
	bool ret;
	char tmp_dir[200];

	// F:/plys/bunny/bun_zipper.ply
	// d:/bunny/bun_zipper.ply
	// F:/plys/happy_recon/happy_vrip.ply

	filename = "F:/plys/bunny/bun_zipper.ply";
	stringToCstr(getFilename(filename).c_str() + "_patches", tmp_dir);

	flog << "\t#" << getTime();

	mesh_divide.tmpBase(tmp_dir);
	ret = mesh_divide.readPlyFirst();
	cout << mesh_divide.info();
	flog << mesh_divide.info();
	if (!ret)
		return EXIT_FAILURE;

	mesh_divide.clearInfo();
	ret = mesh_divide.readPlySecond(3, 3, 3);
	cout << mesh_divide.info();
	flog << mesh_divide.info();
	if (!ret)
		return EXIT_FAILURE;

	return EXIT_SUCCESS;
}
