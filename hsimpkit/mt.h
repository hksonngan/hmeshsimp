/*
   Marching Tetrahedron Implementation
   from http://paulbourke.net/geometry/polygonise/
 */

#ifndef __MARCHING_TETRAHEDRON_H_ISO_
#define __MARCHING_TETRAHEDRON_H_ISO_

#include "mc.h"

namespace MT {
	typedef struct _Tetra {
		XYZ p[4];
		float val[4];
	} Tetra;

	int PolygoniseTri(Tetra &g, const float &iso, float *tri);
	void PolygoniseTriGetCount(const Tetra &g, const float &iso, unsigned char &ntri);
}

#endif //__MARCHING_TETRAHEDRON_H_ISO_