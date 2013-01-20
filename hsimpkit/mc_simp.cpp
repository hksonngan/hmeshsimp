#include "mc_simp.h"

MCSimp::MCSimp(
	float _decimateRate, 
	HTriple<uint> _sliceCount,
	HVertex _cubeLen, 
	HVertex *_pMCCoordStart):
 decimateRate(_decimateRate),
 sliceCount(_sliceCount),
 cubeLen(_cubeLen) {
	pcol = new QuadricEdgeCollapse();
	if (_pMCCoordStart) 
		MCCoordStart = *_pMCCoordStart;
	else
		MCCoordStart.Set(0, 0, 0);
}

MCSimp::~MCSimp() {
	delete pcol;
}

bool MCSimp::addTriangles(
	  Byte *data, 
	  uint nTri, 
	  uint vertSize,  
	  uint vertCoordFirstDimOffSet, 
	  uint vertCoordSecondDimOffSet, 
	  uint vertCoordThirdDimOffSet,
	  DATA_TYPE coordDataType) {

	HVertex vert1, vert2, vert3;
	DataType dataType(coordDataType);

	for (int i = 0; i < nTri; i ++) {
		getVert(vert1, dataType, data, vertCoordFirstDimOffSet, 
			vertCoordSecondDimOffSet, vertCoordThirdDimOffSet);
		data += vertSize;
		getVert(vert1, dataType, data, vertCoordFirstDimOffSet, 
			vertCoordSecondDimOffSet, vertCoordThirdDimOffSet);
		data += vertSize;
		getVert(vert1, dataType, data, vertCoordFirstDimOffSet, 
			vertCoordSecondDimOffSet, vertCoordThirdDimOffSet);
		data += vertSize;
		
		
	}

	return true;
}

bool MCSimp::genIsosurfaces(string filename, double isovalue, vector<TRIANGLE> &tris) {
	if (!rawSet.parseDataFile(filename))
		return false;

	int ncubetris, i;
	TRIANGLE cubetris[5];
	GRIDCELL cube;

	tris.clear();
	while (rawSet.hasNext()) {
		if (!rawSet.nextCube(cube))
			return false;
		ncubetris = MC::Polygonise(cube, isovalue, cubetris);
		for (i = 0; i < ncubetris; i ++)
			tris.push_back(cubetris[i]);
	}

	return true;
}

/*
   Linearly interpolate the position where an isosurface cuts
   an edge between two vertices, each with their own scalar value
*/
XYZ MCSimp::vertexInterp(double isolevel, XYZ p1, XYZ p2, double valp1, double valp2)
{
   double mu;
   XYZ p;

   if (ABS(isolevel-valp1) < 0.00001)
      return(p1);
   if (ABS(isolevel-valp2) < 0.00001)
      return(p2);
   if (ABS(valp1-valp2) < 0.00001)
      return(p1);

   mu = (isolevel - valp1) / (valp2 - valp1);
   p.x = p1.x + mu * (p2.x - p1.x);
   p.y = p1.y + mu * (p2.y - p1.y);
   p.z = p1.z + mu * (p2.z - p1.z);

   return(p);
}

/*
   Given a grid cell and an isolevel, calculate the triangular
   facets required to represent the isosurface through the cell.
   Return the number of triangular facets, the array "triangles"
   will be loaded up with the vertices at most 5 triangular facets.
	0 will be returned if the grid cell is either totally above
   of totally below the isolevel.
*/
int MCSimp::polygonise(HTriple<uint> cubeIndex, GRIDCELL grid, TRIANGLE *triangles)
{
   int i, ntriang;
   int cubeindex;
   XYZ vertlist[12];
   extern int edgeTable[256];
   extern int triTable[256][16];
   InterpOnWhich onWhich;

   /*
      Determine the index into the edge table which
      tells us which vertices are inside of the surface
   */
   cubeindex = 0;
   if (grid.val[0] < isolevel) cubeindex |= 1;
   if (grid.val[1] < isolevel) cubeindex |= 2;
   if (grid.val[2] < isolevel) cubeindex |= 4;
   if (grid.val[3] < isolevel) cubeindex |= 8;
   if (grid.val[4] < isolevel) cubeindex |= 16;
   if (grid.val[5] < isolevel) cubeindex |= 32;
   if (grid.val[6] < isolevel) cubeindex |= 64;
   if (grid.val[7] < isolevel) cubeindex |= 128;

   /* Cube is entirely in/out of the surface */
   if (edgeTable[cubeindex] == 0)
      return(0);

   /* Find the vertices where the surface intersects the cube */
   if (edgeTable[cubeindex] & 1)
      vertlist[0] =
         vertexInterp(isolevel, grid.p[0], grid.p[1], grid.val[0], grid.val[1]);
   if (edgeTable[cubeindex] & 2)
      vertlist[1] =
         vertexInterp(isolevel, grid.p[1], grid.p[2], grid.val[1], grid.val[2]);
   if (edgeTable[cubeindex] & 4)
      vertlist[2] =
         vertexInterp(isolevel, grid.p[2], grid.p[3], grid.val[2], grid.val[3]);
   if (edgeTable[cubeindex] & 8)
      vertlist[3] =
         vertexInterp(isolevel, grid.p[3], grid.p[0], grid.val[3], grid.val[0]);
   if (edgeTable[cubeindex] & 16)
      vertlist[4] =
         vertexInterp(isolevel, grid.p[4], grid.p[5], grid.val[4], grid.val[5]);
   if (edgeTable[cubeindex] & 32)
      vertlist[5] =
         vertexInterp(isolevel, grid.p[5], grid.p[6], grid.val[5], grid.val[6]);
   if (edgeTable[cubeindex] & 64)
      vertlist[6] =
         vertexInterp(isolevel, grid.p[6], grid.p[7], grid.val[6], grid.val[7]);
   if (edgeTable[cubeindex] & 128)
      vertlist[7] =
         vertexInterp(isolevel, grid.p[7], grid.p[4], grid.val[7], grid.val[4]);
   if (edgeTable[cubeindex] & 256)
      vertlist[8] =
         vertexInterp(isolevel, grid.p[0], grid.p[4], grid.val[0], grid.val[4]);
   if (edgeTable[cubeindex] & 512)
      vertlist[9] =
         vertexInterp(isolevel, grid.p[1], grid.p[5], grid.val[1], grid.val[5]);
   if (edgeTable[cubeindex] & 1024)
      vertlist[10] =
         vertexInterp(isolevel, grid.p[2], grid.p[6], grid.val[2], grid.val[6]);
   if (edgeTable[cubeindex] & 2048)
      vertlist[11] =
         vertexInterp(isolevel, grid.p[3], grid.p[7], grid.val[3], grid.val[7]);

   /* Create the triangle */
   ntriang = 0;
   for (i=0;triTable[cubeindex][i]!=-1;i+=3) {
      triangles[ntriang].p[0] = vertlist[triTable[cubeindex][i  ]];
      triangles[ntriang].p[1] = vertlist[triTable[cubeindex][i+1]];
      triangles[ntriang].p[2] = vertlist[triTable[cubeindex][i+2]];
      ntriang++;
   }

   return(ntriang);
}