#include <iostream>
#include <fstream>

#include "divide_grid_mesh.h"

using std::ofstream;
using std::cout;

int main(int argc, char** argv)
{
	HMeshGridDivide mesh_divide;

	// d:/bunny/bun_zipper.ply

	mesh_divide.tmpBase("bunny_patches");
	mesh_divide.readPlyFirst("d:/bunny/bun_zipper.ply");
	cout << mesh_divide.info();

	mesh_divide.readPlySecond(20, 20, 20);
	cout << mesh_divide.info();

	return 0;
}

