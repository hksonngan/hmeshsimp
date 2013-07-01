/*
   Marching Cubes Implementation
   from http://paulbourke.net/geometry/polygonise/
 */

#ifndef __MC_H__
#define __MC_H__

//#define ABS(v) ((v)>=0? (v): -(v))

//typedef struct _XYZ {
//	double x, y, z;
//
//	bool operator == (struct _XYZ &v) {
//		return x == v.x && y == v.y && z == v.z;
//	}
//} XYZ;

typedef struct {
	float x,y,z;

	void set(float a, float b, float c) {
		x = a;
		y = b;
		z = c;
	}
} XYZ;

typedef struct {
	XYZ p[3];
} TRIANGLE;

typedef struct {
	XYZ p[8];
	double val[8];
} GRIDCELL;

namespace MC {

extern int edgeTable[256];
extern int triTable[256][16];

int Polygonise(GRIDCELL grid, double isolevel, TRIANGLE *triangles);

}

#endif