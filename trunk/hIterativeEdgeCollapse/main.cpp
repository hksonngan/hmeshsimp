/*
 *  iterative edge collapse
 *
 *  author: ht
 *  email : waytofall916@gmail.com
 */

#include "ecol_iterative_quadric.h"

int main(int argc, char** argv) {
	
	QuadricEdgeCollapse qec;

	//d:/dragon_recon/dragon_vrip.ply
	//d:/bun_zipper.ply
	
	if (qec.readPly("d:/bun_zipper.ply"))
		return 1;

	if (qec.targetFace(5000))
		return 1;

	return 0;
}

