#include "mc_simp.h"

MCSimp::MCSimp(double _initDecimateRate): 
	initDecimateRate(_initDecimateRate),
	pcol(NULL),
	INFO("") {
}

MCSimp::~MCSimp() {
	if (pcol)
		delete pcol;
}

bool MCSimp::genIsosurfaces(string filename, double _isovalue, vector<TRIANGLE> &tris) { 
	HAugTime htime;
	clearInfo();

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

	ostringstream oss;
	oss << "#iso surfaces generated" << endl
		<< "#volume set resolution: " << volSet.volumeSize.s[0] << "x" << volSet.volumeSize.s[1] << "x" << volSet.volumeSize.s[2] << endl
		<< "#generated faces: " << tris.size() << endl 
		<< "#time consuming: " << htime.printElapseSec() << endl << endl;
	addInfo(oss.str());

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
void MCSimp::polygonise(const UINT4& gridIndex, const GRIDCELL& grid)
{
   int i, ntriang;
   int cubeindex;
   using MC::edgeTable;
   using MC::edgeTable;
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
      return;

   for (i = 0; i < 12; i ++)
	   onWhich[i] = None;

   /* Find the vertices where the surface intersects the cube */
   if (edgeTable[cubeindex] & 1) {
      v = vertexInterp(grid.p[0], grid.p[1], grid.val[0], grid.val[1], onWhich[0]);
	  vertlist[0].Set(v.x, v.y, v.z);
	  vertIndex[0] = getVertIndex(vertlist[0]);
   }
   if (edgeTable[cubeindex] & 2) {
      v = vertexInterp(grid.p[1], grid.p[2], grid.val[1], grid.val[2], onWhich[1]);
	  vertlist[1].Set(v.x, v.y, v.z);
	  vertIndex[1] = getVertIndex(vertlist[1]);
   }
   if (edgeTable[cubeindex] & 4) {
      v = vertexInterp(grid.p[2], grid.p[3], grid.val[2], grid.val[3], onWhich[2]);
	  vertlist[2].Set(v.x, v.y, v.z);
	  vertIndex[2] = getVertIndex(vertlist[2]);
   }
   if (edgeTable[cubeindex] & 8) {
      v = vertexInterp(grid.p[3], grid.p[0], grid.val[3], grid.val[0], onWhich[3]);
	  vertlist[3].Set(v.x, v.y, v.z);
	  vertIndex[3] = getVertIndex(vertlist[3]);
   }
   if (edgeTable[cubeindex] & 16) {
      v = vertexInterp(grid.p[4], grid.p[5], grid.val[4], grid.val[5], onWhich[4]);
	  vertlist[4].Set(v.x, v.y, v.z);
	  vertIndex[4] = getVertIndex(vertlist[4]);
   }
   if (edgeTable[cubeindex] & 32) {
      v = vertexInterp(grid.p[5], grid.p[6], grid.val[5], grid.val[6], onWhich[5]);
	  vertlist[5].Set(v.x, v.y, v.z);
	  vertIndex[5] = getVertIndex(vertlist[5]);
   }
   if (edgeTable[cubeindex] & 64) {
      v = vertexInterp(grid.p[6], grid.p[7], grid.val[6], grid.val[7], onWhich[6]);
	  vertlist[6].Set(v.x, v.y, v.z);
	  vertIndex[6] = getVertIndex(vertlist[6]);
   }
   if (edgeTable[cubeindex] & 128) {
      v = vertexInterp(grid.p[7], grid.p[4], grid.val[7], grid.val[4], onWhich[7]);
	  vertlist[7].Set(v.x, v.y, v.z);
	  vertIndex[7] = getVertIndex(vertlist[7]);
   }
   if (edgeTable[cubeindex] & 256) {
      v = vertexInterp(grid.p[0], grid.p[4], grid.val[0], grid.val[4], onWhich[8]);
	  vertlist[8].Set(v.x, v.y, v.z);
	  vertIndex[8] = getVertIndex(vertlist[8]);
   }
   if (edgeTable[cubeindex] & 512) {
      v = vertexInterp(grid.p[1], grid.p[5], grid.val[1], grid.val[5], onWhich[9]);
	  vertlist[9].Set(v.x, v.y, v.z);
	  vertIndex[9] = getVertIndex(vertlist[9]);
   }
   if (edgeTable[cubeindex] & 1024) {
      v = vertexInterp(grid.p[2], grid.p[6], grid.val[2], grid.val[6], onWhich[10]);
	  vertlist[10].Set(v.x, v.y, v.z);
	  vertIndex[10] = getVertIndex(vertlist[10]);
   }
   if (edgeTable[cubeindex] & 2048) {
      v = vertexInterp(grid.p[3], grid.p[7], grid.val[3], grid.val[7], onWhich[11]);
	  vertlist[11].Set(v.x, v.y, v.z);
	  vertIndex[11] = getVertIndex(vertlist[11]);
   }

   HFace face;
   /* Create the triangle */
   for (i = 0; triTable[cubeindex][i] != -1; i += 3) {
       face.i = vertIndex[triTable[cubeindex][i]];
       face.j = vertIndex[triTable[cubeindex][i+1]];
       face.k = vertIndex[triTable[cubeindex][i+2]];
	   if (pcol->addFace(genFaceCount, face)) {
		   genFaceCount ++;
		   newFaceCount ++;
	   }
   }

   // finalize vertex
   if (edgeTable[cubeindex] & 1) {
	   if (downMost(gridIndex) && (onWhich[0] != Vert2 || rightMost(gridIndex))) {
	       // finalize it   
		   finalizeVert(vertIndex[0], vertlist[0]);
	   }
   }
   if (edgeTable[cubeindex] & 2) {
	   if (rightDownMost(gridIndex) && (onWhich[1] != Vert2 || backMost(gridIndex))) {
	       // finalize it
		   finalizeVert(vertIndex[1], vertlist[1]);
	   }
   }
   if (edgeTable[cubeindex] & 4) {
	   if (backDownMost(gridIndex) && (onWhich[2] != Vert1 || rightMost(gridIndex))) {
		   // finalize it
		   finalizeVert(vertIndex[2], vertlist[2]);
	   }
   }
   if (edgeTable[cubeindex] & 8) {
	   if (downMost(gridIndex) && (onWhich[3] != Vert1 || backMost(gridIndex))) {
		   // finalize it
		   finalizeVert(vertIndex[3], vertlist[3]);
	   }
   }
   if (edgeTable[cubeindex] & 16) {
	   if (onWhich[4] != Vert2 || rightMost(gridIndex)) {
		   // finalize it
		   finalizeVert(vertIndex[4], vertlist[4]);
	   }
   }
   if (edgeTable[cubeindex] & 32) {
	   if (rightMost(gridIndex) && (onWhich[5] != Vert2 || backMost(gridIndex))) {
		   // finalize it
		   finalizeVert(vertIndex[5], vertlist[5]);
	   }
   }
   if (edgeTable[cubeindex] & 64) {
	   if (backMost(gridIndex) && (onWhich[6] != Vert1 || rightMost(gridIndex))) {
		   // finalize it
		   finalizeVert(vertIndex[6], vertlist[6]);
	   }
   }
   if (edgeTable[cubeindex] & 128) {
	   if (onWhich[7] != Vert1 || backMost(gridIndex)) {
		   // finalize it
		   finalizeVert(vertIndex[7], vertlist[7]);
	   }
   }
   if (edgeTable[cubeindex] & 256) {
	   if (onWhich[8] != Vert1 || downMost(gridIndex)) {
		   // finalize it
		   finalizeVert(vertIndex[8], vertlist[8]);
	   }
   }
   if (edgeTable[cubeindex] & 512) {
	   if (rightMost(gridIndex) && (onWhich[9] != Vert1 || downMost(gridIndex))) {
		   // finalize it
		   finalizeVert(vertIndex[9], vertlist[9]);
	   }
   }
   if (edgeTable[cubeindex] & 1024) {
	   if (rightBackMost(gridIndex) && (onWhich[10] != Vert1 || downMost(gridIndex))) {
		   // finalize it
		   finalizeVert(vertIndex[10], vertlist[10]);
	   }
   }
   if (edgeTable[cubeindex] & 2048) {
	   if (backMost(gridIndex) && (onWhich[11] != Vert1 || downMost(gridIndex))) {
		   // finalize it1
		   finalizeVert(vertIndex[11], vertlist[11]);
	   }
   }
}

void MCSimp::finalizeVert(const uint &index, const HVertex &v) {
	CollapsableVertex &cv = pcol->v(index);
	if (cv.mark == FINAL)
		return;

	// first erase from the vertex map
	vertexMap.erase(v);

	// add pairs
	vert_arr starVertices;
	face_arr _faces;
	pcol->collectStarVertices(index, &starVertices);
	for (int j = 0; j < starVertices.count(); j ++) {
		CollapsableVertex &cv2 = pcol->v(starVertices[j]);
		// add specific edge only once
		// never collapse the exterior vertices
		if (cv2.mark == FINAL) {
			CollapsablePair *new_pair = new CollapsablePair(index, starVertices[j]);
			new_pair->keepOrder();
			pcol->addCollapsablePair(new_pair);
		}
	}

	cv.markv(FINAL);
}

bool MCSimp::genCollapse(
		string filename, double _isovalue, double decimateRate, 
		unsigned int maxNewTri, unsigned int &nvert, unsigned int &nface) 
{
	HAugTime htime;
	clearInfo();

	if (decimateRate <= 0 || decimateRate >= 1)
		return false;
	isovalue = _isovalue;
	genFaceCount = 0;
	newFaceCount = 0;
	genVertCount = 0;

	if (!volSet.parseDataFile(filename))
		return false;

	if (pcol)
		delete pcol;
	pcol = new QuadricEdgeCollapse();
	UINT4 cubeIndex;
	GRIDCELL cube;

	// init decimation
	if (decimateRate < initDecimateRate) {
		// first read in maxNewTri triangles and decimate based on initDecimateRate
		while (volSet.hasNext()) {
			cubeIndex = volSet.cursor;

			if (!volSet.nextCube(cube))
				return false;
			polygonise(cubeIndex, cube);

			if (newFaceCount >= maxNewTri - 2) {
				pcol->targetFace(pcol->validFaces() * initDecimateRate);
				newFaceCount = 0;
				break;
			}
		}

		// than read in maxNewTri * (1 - initDecimateRate) triangles and decimate  
		// til the triangles left equal to maxNewTri * initDecimateRate.
		// The outer loop stops when the true decimate rate will be lower than
		// the given decimate rate next time.
		unsigned int initReadCount;
		while (true) {
			// approximated decimate rate of this iteration is
			// lower than the given decimate rate
			initReadCount = maxNewTri - pcol->validFaces();
			if (maxNewTri * initDecimateRate / (genFaceCount + initReadCount) < decimateRate)
				break;

			while (volSet.hasNext()) {
				cubeIndex = volSet.cursor;
				if (!volSet.nextCube(cube))
					return false;
				polygonise(cubeIndex, cube);

				if (pcol->validFaces() >= maxNewTri - 4) {
					pcol->targetFace(pcol->validFaces() * initDecimateRate);
					newFaceCount = 0;
					break;
				}
			}
		}
	}

	while (volSet.hasNext()) {
		cubeIndex = volSet.cursor;
		if (!volSet.nextCube(cube))
			return false;
		polygonise(cubeIndex, cube);

		if (newFaceCount >= maxNewTri - 2) {
			pcol->targetFace(pcol->validFaces() - newFaceCount + newFaceCount * decimateRate);
			newFaceCount = 0;
		}
	}

	pcol->targetFace(genFaceCount * decimateRate);
	
	nvert = pcol->validVerts();
	nface = pcol->validFaces();

	ostringstream oss;
	oss << "#iso surfaces decimated" << endl
		<< "#volume set resolution: " << volSet.volumeSize.s[0] << "x" << volSet.volumeSize.s[1] << "x" << volSet.volumeSize.s[2] << endl
		<< "#generated faces: " << genFaceCount << ", vertices: " << genVertCount << endl
		<< "#simplified faces: " << nface << ", vertices: " << nvert << endl 
		<< "#time consuming: " << htime.printElapseSec() << endl << endl;
	addInfo(oss.str());

	return true;
}

void MCSimp::toIndexedMesh(HVertex *vertArr, HFace *faceArr) {
	pcol->toIndexedMesh(vertArr, faceArr);
}