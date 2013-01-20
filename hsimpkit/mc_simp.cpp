#include "mc_simp.h"

//MCSimp::MCSimp(
//	float _decimateRate, 
//	HTriple<uint> _sliceCount,
//	HVertex _cubeLen, 
//	HVertex *_pMCCoordStart):
// decimateRate(_decimateRate),
// sliceCount(_sliceCount),
// cubeLen(_cubeLen) {
//	pcol = new QuadricEdgeCollapse();
//	if (_pMCCoordStart) 
//		MCCoordStart = *_pMCCoordStart;
//	else
//		MCCoordStart.Set(0, 0, 0);
//}

MCSimp::~MCSimp() {
	if (pcol)
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

bool MCSimp::genIsosurfaces(string filename, double _isovalue, vector<TRIANGLE> &tris) 

	if (!volSet.parseDataFile(filename))
		return false;

	int ncubetris, i;
	TRIANGLE cubetris[5];
	GRIDCELL cube;

	tris.clear();
	while (volSet.hasNext()) {
		if (!volSet.nextCube(cube))
			return false;
		ncubetris = MC::Polygonise(cube, _isovalue, cubetris);
		for (i = 0; i < ncubetris; i ++)
			tris.push_back(cubetris[i]);
	}

	return true;
}

/*
   Linearly interpolate the position where an isosurface cuts
   an edge between two vertices, each with their own scalar value
*/
XYZ MCSimp::vertexInterp(XYZ p1, XYZ p2, double valp1, double valp2, InterpOnWhich& onWhich)
{
   double mu;
   XYZ p;

   if (ABS(isovalue-valp1) < 0.00001) {
      onWhich = Vert1;
      return(p1);
   }
   if (ABS(isovalue-valp2) < 0.00001) {
      onWhich = Vert2;
	  return(p2);
   }
   if (ABS(valp1-valp2) < 0.00001) {
      onWhich = Vert1;
	  return(p1);
   }

   mu = (isovalue - valp1) / (valp2 - valp1);
   p.x = p1.x + mu * (p2.x - p1.x);
   p.y = p1.y + mu * (p2.y - p1.y);
   p.z = p1.z + mu * (p2.z - p1.z);

   onWhich = Edge;
   return(p);
}

/*
   Given a grid cell and an isovalue, calculate the triangular
   facets required to represent the isosurface through the cell.
   Return the number of triangular facets, the array "triangles"
   will be loaded up with the vertices at most 5 triangular facets.
	0 will be returned if the grid cell is either totally above
   of totally below the isovalue.
*/
void MCSimp::polygonise(const FLOAT4& gridIndex, const GRIDCELL& grid)
{
   int i, ntriang;
   int cubeindex;
   extern int edgeTable[256];
   extern int triTable[256][16];
   XYZ v;

   /*
      Determine the index into the edge table which
      tells us which vertices are inside of the surface
   */
   cubeindex = 0;
   if (grid.val[0] < isovalue) cubeindex |= 1;
   if (grid.val[1] < isovalue) cubeindex |= 2;
   if (grid.val[2] < isovalue) cubeindex |= 4;
   if (grid.val[3] < isovalue) cubeindex |= 8;
   if (grid.val[4] < isovalue) cubeindex |= 16;
   if (grid.val[5] < isovalue) cubeindex |= 32;
   if (grid.val[6] < isovalue) cubeindex |= 64;
   if (grid.val[7] < isovalue) cubeindex |= 128;

   /* Cube is entirely in/out of the surface */
   if (edgeTable[cubeindex] == 0)
      return(0);

   for (i = 0; i < 12; i ++)
	   onWhich[i] = None;

   /* Find the vertices where the surface intersects the cube */
   if (edgeTable[cubeindex] & 1) {
      v = vertexInterp(grid.p[0], grid.p[1], grid.val[0], grid.val[1], &onWhich[0]);
	  vertlist[0].Set(v.x, v.y, v.z);
	  vertIndex[0] = getVertIndex(vertlist[0]);
   }
   if (edgeTable[cubeindex] & 2) {
      v = vertexInterp(grid.p[1], grid.p[2], grid.val[1], grid.val[2], &onWhich[1]);
	  vertlist[1].Set(v.x, v.y, v.z);
	  vertIndex[1] = getVertIndex(vertlist[1]);
   }
   if (edgeTable[cubeindex] & 4) {
      v = vertexInterp(grid.p[2], grid.p[3], grid.val[2], grid.val[3], &onWhich[2]);
	  vertlist[2].Set(v.x, v.y, v.z);
	  vertIndex[2] = getVertIndex(vertlist[2]);
   }
   if (edgeTable[cubeindex] & 8) {
      v = vertexInterp(grid.p[3], grid.p[0], grid.val[3], grid.val[0], &onWhich[3]);
	  vertlist[3].Set(v.x, v.y, v.z);
	  vertIndex[3] = getVertIndex(vertlist[3]);
   }
   if (edgeTable[cubeindex] & 16) {
      v = vertexInterp(grid.p[4], grid.p[5], grid.val[4], grid.val[5], &onWhich[4]);
	  vertlist[4].Set(v.x, v.y, v.z);
	  vertIndex[4] = getVertIndex(vertlist[4]);
   }
   if (edgeTable[cubeindex] & 32) {
      v = vertexInterp(grid.p[5], grid.p[6], grid.val[5], grid.val[6], &onWhich[5]);
	  vertlist[5].Set(v.x, v.y, v.z);
	  vertIndex[5] = getVertIndex(vertlist[5]);
   }
   if (edgeTable[cubeindex] & 64) {
      v = vertexInterp(grid.p[6], grid.p[7], grid.val[6], grid.val[7], &onWhich[6]);
	  vertlist[6].Set(v.x, v.y, v.z);
	  vertIndex[6] = getVertIndex(vertlist[6]);
   }
   if (edgeTable[cubeindex] & 128) {
      v = vertexInterp(grid.p[7], grid.p[4], grid.val[7], grid.val[4], &onWhich[7]);
	  vertlist[7].Set(v.x, v.y, v.z);
	  vertIndex[7] = getVertIndex(vertlist[7]);
   }
   if (edgeTable[cubeindex] & 256) {
      v = vertexInterp(grid.p[0], grid.p[4], grid.val[0], grid.val[4], &onWhich[8]);
	  vertlist[8].Set(v.x, v.y, v.z);
	  vertIndex[8] = getVertIndex(vertlist[8]);
   }
   if (edgeTable[cubeindex] & 512) {
      v = vertexInterp(grid.p[1], grid.p[5], grid.val[1], grid.val[5], &onWhich[9]);
	  vertlist[9].Set(v.x, v.y, v.z);
	  vertIndex[9] = getVertIndex(vertlist[9]);
   }
   if (edgeTable[cubeindex] & 1024) {
      v = vertexInterp(grid.p[2], grid.p[6], grid.val[2], grid.val[6], &onWhich[10]);
	  vertlist[10].Set(v.x, v.y, v.z);
	  vertIndex[10] = getVertIndex(vertlist[10]);
   }
   if (edgeTable[cubeindex] & 2048) {
      v = vertexInterp(grid.p[3], grid.p[7], grid.val[3], grid.val[7], &onWhich[11]);
	  vertlist[11].Set(v.x, v.y, v.z);
	  vertIndex[11] = getVertIndex(vertlist[11]);
   }

   HFace face;
   /* Create the triangle */
   for (i = 0; triTable[cubeindex][i] != -1; i += 3) {
       face.i = vertIndex[triTable[cubeindex][i]];
       face.j = vertIndex[triTable[cubeindex][i+1]];
       face.k = vertIndex[triTable[cubeindex][i+2]];
	   pcol->addFace(face);
   }

   // finalize vertex
   if (edgeTable[cubeindex] & 1) {
	   v = vertexInterp(grid.p[0], grid.p[1], grid.val[0], grid.val[1], &onWhich[0]);
	   vertlist[0].Set(v.x, v.y, v.z);
	   vertIndex[0] = getVertIndex(vertlist[0]);

	   if (downMost(cubeindex) && (onWhich != Vert2 || rightMost(cubeindex))) {
	       // finalize it   
	   }
   }
   if (edgeTable[cubeindex] & 2) {
	   v = vertexInterp(grid.p[1], grid.p[2], grid.val[1], grid.val[2], &onWhich[1]);
	   vertlist[1].Set(v.x, v.y, v.z);
	   vertIndex[1] = getVertIndex(vertlist[1]);

	   if (rightDownMost(cubeindex) && (onWhich != Vert2 || backMost(cubeindex))) {
	       // finalize it
	   }
   }
   if (edgeTable[cubeindex] & 4) {
	   v = vertexInterp(grid.p[2], grid.p[3], grid.val[2], grid.val[3], &onWhich[2]);
	   vertlist[2].Set(v.x, v.y, v.z);
	   vertIndex[2] = getVertIndex(vertlist[2]);

	   if (backDownMost(cubeindex) && (onWhich != Vert1 || rightMost(cubeindex))) {
		   // finalize it
	   }
   }
   if (edgeTable[cubeindex] & 8) {
	   v = vertexInterp(grid.p[3], grid.p[0], grid.val[3], grid.val[0], &onWhich[3]);
	   vertlist[3].Set(v.x, v.y, v.z);
	   vertIndex[3] = getVertIndex(vertlist[3]);

	   if (downMost(cubeindex) && (onWhich != Vert1 || backMost(cubeindex))) {
		   // finalize it
	   }
   }
   if (edgeTable[cubeindex] & 16) {
	   v = vertexInterp(grid.p[4], grid.p[5], grid.val[4], grid.val[5], &onWhich[4]);
	   vertlist[4].Set(v.x, v.y, v.z);
	   vertIndex[4] = getVertIndex(vertlist[4]);

	   if (onWhich != Vert2 || rightMost(cubeindex)) {
		   // finalize it
	   }
   }
   if (edgeTable[cubeindex] & 32) {
	   v = vertexInterp(grid.p[5], grid.p[6], grid.val[5], grid.val[6], &onWhich[5]);
	   vertlist[5].Set(v.x, v.y, v.z);
	   vertIndex[5] = getVertIndex(vertlist[5]);

	   if (rightMost(cubeindex) && (onWhich != Vert2 || backMost(cubeindex))) {
		   // finalize it
	   }
   }
   if (edgeTable[cubeindex] & 64) {
	   v = vertexInterp(grid.p[6], grid.p[7], grid.val[6], grid.val[7], &onWhich[6]);
	   vertlist[6].Set(v.x, v.y, v.z);
	   vertIndex[6] = getVertIndex(vertlist[6]);

	   if (backMost(cubeindex) && (onWhich != Vert1 || rightMost(cubeindex))) {
		   // finalize it
	   }
   }
   if (edgeTable[cubeindex] & 128) {
	   v = vertexInterp(grid.p[7], grid.p[4], grid.val[7], grid.val[4], &onWhich[7]);
	   vertlist[7].Set(v.x, v.y, v.z);
	   vertIndex[7] = getVertIndex(vertlist[7]);

	   if (onWhich != Vert1 || backMost(cubeindex)) {
		   // finalize it
	   }
   }
   if (edgeTable[cubeindex] & 256) {
	   v = vertexInterp(grid.p[0], grid.p[4], grid.val[0], grid.val[4], &onWhich[8]);
	   vertlist[8].Set(v.x, v.y, v.z);
	   vertIndex[8] = getVertIndex(vertlist[8]);

	   if (onWhich != Vert1 || downMost(cubeindex)) {
		   // finalize it
	   }
   }
   if (edgeTable[cubeindex] & 512) {
	   v = vertexInterp(grid.p[1], grid.p[5], grid.val[1], grid.val[5], &onWhich[9]);
	   vertlist[9].Set(v.x, v.y, v.z);
	   vertIndex[9] = getVertIndex(vertlist[9]);

	   if (rightMost(cubeindex) && (onWhich != Vert1 || downMost(cubeindex))) {
		   // finalize it
	   }
   }
   if (edgeTable[cubeindex] & 1024) {
	   v = vertexInterp(grid.p[2], grid.p[6], grid.val[2], grid.val[6], &onWhich[10]);
	   vertlist[10].Set(v.x, v.y, v.z);
	   vertIndex[10] = getVertIndex(vertlist[10]);

	   if (rightBackMost(cubeindex) && (onWhich != Vert1 || downMost(cubeindex))) {
		   // finalize it
	   }
   }
   if (edgeTable[cubeindex] & 2048) {
	   v = vertexInterp(grid.p[3], grid.p[7], grid.val[3], grid.val[7], &onWhich[11]);
	   vertlist[11].Set(v.x, v.y, v.z);
	   vertIndex[11] = getVertIndex(vertlist[11]);

	   if (backMost(cubeindex) && (onWhich != Vert1 || downMost(cubeindex))) {
		   // finalize it
	   }
   }
}

bool MCSimp::genCollapse(string filename, double _isovalue, double decimateRate) {
	isovalue = _isovalue;	
	if (decimateRate <= 0 || decimateRate >= 1)
		return false;

	FLOAT4 cubeIndex;
	if (!volSet.parseDataFile(filename))
		return false;

	if (pcol)
		delete pcol;
	pcol = new QuadricEdgeCollapse();
	int ncubetris, i;
	TRIANGLE cubetris[5];
	GRIDCELL cube;
	vertCount = 0;

	while (volSet.hasNext()) {
		cubeIndex = volSet.cursor;
		if (!volSet.nextCube(cube))
			return false;
		ncubetris = polygonise(cubeIndex, cube, cubetris);
		for (i = 0; i < ncubetris; i ++)
			cubetris[i];
	}

	delete pcol;
	pcol = NULL;
	return true;
}