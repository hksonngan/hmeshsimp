#include <iostream>
#include <fstream>

#include "divide_grid_mesh.h"

using std::ofstream;

int main(int argc, char** argv)
{
	HMeshGridDivide mesh_divide;

	// d:/bunny/bun_zipper.ply

	mesh_divide.readPlyFirst("d:/bunny/bun_zipper.ply");
	mesh_divide.readPlySecond(20, 20, 20);

	return 0;
}

