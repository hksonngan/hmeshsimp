#include <iostream>
#include <fstream>

#include "patching_simp.h"
#include "trivial.h"
#include "mem_stream.h"

using std::ofstream;
using std::fstream;
using std::cout;
using std::endl;

static char* filename;
static uint x_div = 2, y_div = 2, z_div = 2;
static uint target = 4000;

int main(int argc, char** argv)
{
	mstream<int> m;

	m << 1 << 2 << 3;

	return 0;

	PatchingSimp psimp;
	ofstream flog("patchingsimp.log", fstream::app);
	bool ret;
	char tmp_dir[200];

	// F:/plys/bunny/bun_zipper.ply
	// d:/bunny/bun_zipper.ply
	// F:/plys/happy_recon/happy_vrip.ply

	filename = "d:/bunny/bun_zipper.ply";
	stringToCstr(getFilename(filename) + "_patches", tmp_dir);

	flog << "\t#" << getTime();

	psimp.tmpBase(tmp_dir);
	ret = psimp.readPlyFirst(filename);
	cout << psimp.info();
	flog << psimp.info();
	if (!ret)
		return EXIT_FAILURE;

	psimp.clearInfo();
	ret = psimp.readPlySecond(x_div, y_div, z_div);
	cout << psimp.info();
	flog << psimp.info();
	if (!ret)
		return EXIT_FAILURE;

	psimp.patchesToPly();
	psimp.simplfiyPatchesToPly(target);

	return EXIT_SUCCESS;
}
