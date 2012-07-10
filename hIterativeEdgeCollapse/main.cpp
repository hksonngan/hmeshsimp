/*
 *  iterative edge collapse
 *
 *  author: ht
 *  email : waytofall916@gmail.com
 */

#include "ecol_iterative_quadric.h"

int main(int argc, char** argv) {
	
	QuadricEdgeCollapse qec;

	//F:/plys/dragon_recon/dragon_vrip.ply
	//F:/plys/bunny\bun_zipper.ply
	//F:/plys/happy_recon/happy_vrip.ply
	//F:/plys/Armadillo.ply
	//F:/plys/horse.ply
	//d:/dragon_recon/dragon_vrip.ply
	//d:/bun_zipper.ply
	
	if (!qec.readPly("F:/plys/bunny/bun_zipper_res3.ply"))
		return 1;

	if (!qec.targetFace(500))
		return 1;

	return 0;
}

