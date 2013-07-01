/*
 *  Defines an entry for Cutting Based Simplification
 *
 *  Author: Ht
 *  Email : waytofall916 at gmail dot com
 *
 *  Copyright (C) Ht. All rights reserved.
 */

#include <iostream>
#include <fstream>

#include "patching_simp.h"
#include "trivial.h"

using std::ofstream;
using std::fstream;
using std::cout;
using std::endl;

int psimp_entry(
	char *filename, uint target, uint x_div, uint y_div, uint z_div, bool binary
){
	PatchingSimp psimp;
	ofstream flog("psimp.log", fstream::app);
	bool ret;
	char tmp_dir[200];

	stringToCstr(getFilename(filename) + "_patches", tmp_dir);

	flog << endl << endl << 
		"\t###############################################" << endl 
		<< "\t" << getTime()
		<< "\t" << getExtFilename(filename) << endl;

	psimp.tmpBase(tmp_dir);
	ret = psimp.readPlyFirst(filename);
	if (!ret)
		goto termin;

	ret = psimp.readPlySecond(x_div, y_div, z_div);
	if (!ret)
		goto termin;

	//psimp.patchesToPly();
	//psimp.simplfiyPatchesToPly(target);

	ret = psimp.mergeSimpPly(target, binary);
	if (!ret)
		goto termin;

	termin:
	flog << psimp.info();

	if (ret)
		return EXIT_SUCCESS;
	else
		return EXIT_FAILURE;
}
