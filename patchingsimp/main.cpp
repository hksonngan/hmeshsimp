#include <iostream>
#include <fstream>

#include "patching_simp.h"
#include "trivial.h"
#include "mem_stream.h"
#include "h_algorithm.h"

using std::ofstream;
using std::fstream;
using std::cout;
using std::endl;

static char* filename;
static uint x_div = 2, y_div = 2, z_div = 2;
static uint target = 20000;

int main(int argc, char** argv)
{
	ostringstream oss;
	oss << "!!!!!!!!";
	oss.clear();
	oss.seekp(0);
	oss << "~~~~~~~~";

	PatchingSimp psimp;
	ofstream flog("patchingsimp.log", fstream::app);
	bool ret;
	char tmp_dir[200];

	// F:/plys/bunny/bun_zipper.ply
	// F:/plys/happy_recon/happy_vrip.ply
	// F:/plys/xyzrgb_statuette.ply

	// d:/bunny/bun_zipper.ply
	// d:/happy_recon/happy_vrip.ply
	// d:/hmeshsimp/patchingsimp/bun_zipper_patches/bun_zipper_psimp_bin.ply
	// d:/xyzrgb_statuette.ply
	// d:/lucy.ply

	filename = "F:/plys/xyzrgb_statuette.ply";
	x_div = 2;
	y_div = 2;
	z_div = 2;
	target = 30000;

	stringToCstr(getFilename(filename) + "_patches", tmp_dir);

	flog << endl << endl << 
		"\t###############################################" << endl 
		<< "\t" << getTime();

	psimp.tmpBase(tmp_dir);
	ret = psimp.readPlyFirst(filename);
	if (!ret)
		goto termin;

	ret = psimp.readPlySecond(x_div, y_div, z_div);
	if (!ret)
		goto termin;

	//psimp.patchesToPly();
	//psimp.simplfiyPatchesToPly(target);

	ret = psimp.mergeSimpPly(target, true);
	if (!ret)
		goto termin;

	termin:
	flog << psimp.info();

	if (ret)
		return EXIT_SUCCESS;
	else
		return EXIT_FAILURE;
}
