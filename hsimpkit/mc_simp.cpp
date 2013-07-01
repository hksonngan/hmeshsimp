#include "mc_simp.h"
#include "mt.h"
#include "trivial.h"
#include "io_common.h"

using std::ios;

MCSimp::MCSimp(double _initDecimateRate): 
	_m_init_decimate_rate(_initDecimateRate),
	_m_pcol(NULL),
	_m_final_vert_hook(NULL),
	_m_info("") {
}

MCSimp::~MCSimp() {
	if (_m_pcol)
		delete _m_pcol;
}

bool MCSimp::genIsosurfaces(
    string filename, double _isovalue, int *sampleStride, 
    vector<float> &tris, VolumeSet *paraVolSet) {
	HAugTime htime;
	clearInfo();

    if (paraVolSet)
        _m_vol_set = *paraVolSet;
    else {
        if (!_m_vol_set.parseDataFile(filename, true, false)) 
            return false; 
    }

    if (sampleStride) {
        _m_vol_set.sampleStride.s[0] = sampleStride[0];
        _m_vol_set.sampleStride.s[1] = sampleStride[1];
        _m_vol_set.sampleStride.s[2] = sampleStride[2];
    }

	int ncubetris, i;
	TRIANGLE cubetris[5];
	GRIDCELL cube;

	tris.clear();
	while (_m_vol_set.hasNext()) {
		if (!_m_vol_set.nextCube(cube))
			return false;
		ncubetris = MC::Polygonise(cube, _isovalue, cubetris);
        for (i = 0; i < ncubetris; i ++) {
			tris.push_back(cubetris[i].p[0].x);
            tris.push_back(cubetris[i].p[0].y);
            tris.push_back(cubetris[i].p[0].z);
            tris.push_back(cubetris[i].p[2].x);
            tris.push_back(cubetris[i].p[2].y);
            tris.push_back(cubetris[i].p[2].z);
            tris.push_back(cubetris[i].p[1].x);
            tris.push_back(cubetris[i].p[1].y);
            tris.push_back(cubetris[i].p[1].z);
        }
	}

	ostringstream oss;
	oss << "#iso surfaces generated" << endl
        << "#iso value: " << _isovalue << endl
        << "#volume set resolution: " << _m_vol_set.volumeSize.s[0] << "x" << _m_vol_set.volumeSize.s[1] << "x" << _m_vol_set.volumeSize.s[2] << endl 
        << "#sample stride: " << _m_vol_set.sampleStride.s[0] << "x" << _m_vol_set.sampleStride.s[1] << "x" << _m_vol_set.sampleStride.s[2] << endl
        << "#generated faces: " << tris.size() / 9 << endl 
        << "#time consuming: " << htime.printElapseSec() << endl << endl;
	addInfo(oss.str());

	return true;
}

bool MCSimp::genIsosurfacesMT(
	string filename, double _isovalue, int *sampleStride, 
	vector<float> &tris, VolumeSet *paraVolSet) {
	HAugTime htime;
	clearInfo();

	if (paraVolSet)
		_m_vol_set = *paraVolSet;
	else {
		if (!_m_vol_set.parseDataFile(filename, true, false)) 
			return false; 
	}

	if (sampleStride) {
		_m_vol_set.sampleStride.s[0] = sampleStride[0];
		_m_vol_set.sampleStride.s[1] = sampleStride[1];
		_m_vol_set.sampleStride.s[2] = sampleStride[2];
	}

	int ntris, i, j;
	unsigned char ntris2;
	float cubetris[18];
	GRIDCELL cube;
	MT::Tetra tetra;

	int tetra_index[6][4] = {
		{0, 2, 7, 3},
		{0, 2, 6, 7},
		{4, 0, 6, 7},
		{6, 0, 1, 2},
		{6, 0, 4, 1},
		{5, 1, 6, 4}
	};

	//PolygoniseTri(grid,iso,triangles,0,2,3,7);
	//PolygoniseTri(grid,iso,triangles,0,2,6,7);
	//PolygoniseTri(grid,iso,triangles,0,4,6,7);
	//PolygoniseTri(grid,iso,triangles,0,6,1,2);
	//PolygoniseTri(grid,iso,triangles,0,6,1,4);
	//PolygoniseTri(grid,iso,triangles,5,6,1,4);

	tris.clear();
	while (_m_vol_set.hasNext()) {
		if (!_m_vol_set.nextCube(cube))
			return false;
		for (j = 0; j < 6; j ++) {
			for (i = 0; i < 4; i ++) {
				tetra.p[i] = cube.p[tetra_index[j][i]];
				tetra.val[i] = cube.val[tetra_index[j][i]];
			}
			ntris = MT::PolygoniseTri(tetra, _isovalue, cubetris);
			MT::PolygoniseTriGetCount(tetra, _isovalue, ntris2);
			if (ntris != ntris2) {
				std::cerr << "error while get tetra triangle count" << std::endl;
			}
			for (i = 0; i < ntris; i ++) {
				tris.push_back(cubetris[i*9]);
				tris.push_back(cubetris[i*9+1]);
				tris.push_back(cubetris[i*9+2]);
				tris.push_back(cubetris[i*9+3]);
				tris.push_back(cubetris[i*9+4]);
				tris.push_back(cubetris[i*9+5]);
				tris.push_back(cubetris[i*9+6]);
				tris.push_back(cubetris[i*9+7]);
				tris.push_back(cubetris[i*9+8]);
			}
		}
	}

	ostringstream oss;
	oss << "#iso surfaces generated" << endl
		<< "#iso value: " << _isovalue << endl
		<< "#volume set resolution: " << _m_vol_set.volumeSize.s[0] << "x" << _m_vol_set.volumeSize.s[1] << "x" << _m_vol_set.volumeSize.s[2] << endl 
		<< "#sample stride: " << _m_vol_set.sampleStride.s[0] << "x" << _m_vol_set.sampleStride.s[1] << "x" << _m_vol_set.sampleStride.s[2] << endl
		<< "#generated faces: " << tris.size() / 9 << endl 
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

   if (ABS(_m_isovalue-valp1) < 0.00001) {
      onWhich = Vert1;
      return(p1);
   }
   if (ABS(_m_isovalue-valp2) < 0.00001) {
      onWhich = Vert2;
	  return(p2);
   }
   if (ABS(valp1-valp2) < 0.00001) {
      onWhich = Vert1;
	  return(p1);
   }

   mu = (_m_isovalue - valp1) / (valp2 - valp1);
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
int MCSimp::polygonise(const UINT4& gridIndex, const GRIDCELL& grid, HFace *face)
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
   if (grid.val[0] < _m_isovalue) cubeindex |= 1;
   if (grid.val[1] < _m_isovalue) cubeindex |= 2;
   if (grid.val[2] < _m_isovalue) cubeindex |= 4;
   if (grid.val[3] < _m_isovalue) cubeindex |= 8;
   if (grid.val[4] < _m_isovalue) cubeindex |= 16;
   if (grid.val[5] < _m_isovalue) cubeindex |= 32;
   if (grid.val[6] < _m_isovalue) cubeindex |= 64;
   if (grid.val[7] < _m_isovalue) cubeindex |= 128;

   /* Cube is entirely in/out of the surface */
   if (edgeTable[cubeindex] == 0)
      return 0;

   for (i = 0; i < 12; i ++)
	   _m_on_which[i] = None;

   /* Find the vertices where the surface intersects the cube */
   if (edgeTable[cubeindex] & 1) {
      v = vertexInterp(grid.p[0], grid.p[1], grid.val[0], grid.val[1], _m_on_which[0]);
	  _m_vert_list[0].Set(v.x, v.y, v.z);
	  _m_vert_index[0] = getVertIndex(_m_vert_list[0]);
   }
   if (edgeTable[cubeindex] & 2) {
      v = vertexInterp(grid.p[1], grid.p[2], grid.val[1], grid.val[2], _m_on_which[1]);
	  _m_vert_list[1].Set(v.x, v.y, v.z);
	  _m_vert_index[1] = getVertIndex(_m_vert_list[1]);
   }
   if (edgeTable[cubeindex] & 4) {
      v = vertexInterp(grid.p[2], grid.p[3], grid.val[2], grid.val[3], _m_on_which[2]);
	  _m_vert_list[2].Set(v.x, v.y, v.z);
	  _m_vert_index[2] = getVertIndex(_m_vert_list[2]);
   }
   if (edgeTable[cubeindex] & 8) {
      v = vertexInterp(grid.p[3], grid.p[0], grid.val[3], grid.val[0], _m_on_which[3]);
	  _m_vert_list[3].Set(v.x, v.y, v.z);
	  _m_vert_index[3] = getVertIndex(_m_vert_list[3]);
   }
   if (edgeTable[cubeindex] & 16) {
      v = vertexInterp(grid.p[4], grid.p[5], grid.val[4], grid.val[5], _m_on_which[4]);
	  _m_vert_list[4].Set(v.x, v.y, v.z);
	  _m_vert_index[4] = getVertIndex(_m_vert_list[4]);
   }
   if (edgeTable[cubeindex] & 32) {
      v = vertexInterp(grid.p[5], grid.p[6], grid.val[5], grid.val[6], _m_on_which[5]);
	  _m_vert_list[5].Set(v.x, v.y, v.z);
	  _m_vert_index[5] = getVertIndex(_m_vert_list[5]);
   }
   if (edgeTable[cubeindex] & 64) {
      v = vertexInterp(grid.p[6], grid.p[7], grid.val[6], grid.val[7], _m_on_which[6]);
	  _m_vert_list[6].Set(v.x, v.y, v.z);
	  _m_vert_index[6] = getVertIndex(_m_vert_list[6]);
   }
   if (edgeTable[cubeindex] & 128) {
      v = vertexInterp(grid.p[7], grid.p[4], grid.val[7], grid.val[4], _m_on_which[7]);
	  _m_vert_list[7].Set(v.x, v.y, v.z);
	  _m_vert_index[7] = getVertIndex(_m_vert_list[7]);
   }
   if (edgeTable[cubeindex] & 256) {
      v = vertexInterp(grid.p[0], grid.p[4], grid.val[0], grid.val[4], _m_on_which[8]);
	  _m_vert_list[8].Set(v.x, v.y, v.z);
	  _m_vert_index[8] = getVertIndex(_m_vert_list[8]);
   }
   if (edgeTable[cubeindex] & 512) {
      v = vertexInterp(grid.p[1], grid.p[5], grid.val[1], grid.val[5], _m_on_which[9]);
	  _m_vert_list[9].Set(v.x, v.y, v.z);
	  _m_vert_index[9] = getVertIndex(_m_vert_list[9]);
   }
   if (edgeTable[cubeindex] & 1024) {
      v = vertexInterp(grid.p[2], grid.p[6], grid.val[2], grid.val[6], _m_on_which[10]);
	  _m_vert_list[10].Set(v.x, v.y, v.z);
	  _m_vert_index[10] = getVertIndex(_m_vert_list[10]);
   }
   if (edgeTable[cubeindex] & 2048) {
      v = vertexInterp(grid.p[3], grid.p[7], grid.val[3], grid.val[7], _m_on_which[11]);
	  _m_vert_list[11].Set(v.x, v.y, v.z);
	  _m_vert_index[11] = getVertIndex(_m_vert_list[11]);
   }

   int n = 0;
   /* Create the triangle */
   for (i = 0; triTable[cubeindex][i] != -1; i += 3) {
       face[n].i = _m_vert_index[triTable[cubeindex][i]];
       face[n].j = _m_vert_index[triTable[cubeindex][i+2]];
       face[n].k = _m_vert_index[triTable[cubeindex][i+1]];

	   if (_m_pcol && _m_pcol->addFace(_m_gen_face_count, face[n])) {
		   _m_gen_face_count ++;
		   _m_new_face_count ++;
	   }

	   n ++;
   }

   // finalize vertex
   if (edgeTable[cubeindex] & 1) {
	   if (downMost(gridIndex) && (_m_on_which[0] != Vert2 || rightMost(gridIndex))) {
	       // finalize it   
		   (this->*(_m_final_vert_hook))(_m_vert_index[0], _m_vert_list[0]);
	   }
   }
   if (edgeTable[cubeindex] & 2) {
	   if (rightDownMost(gridIndex) && (_m_on_which[1] != Vert2 || backMost(gridIndex))) {
	       // finalize it
		   (this->*(_m_final_vert_hook))(_m_vert_index[1], _m_vert_list[1]);
	   }
   }
   if (edgeTable[cubeindex] & 4) {
	   if (backDownMost(gridIndex) && (_m_on_which[2] != Vert1 || rightMost(gridIndex))) {
		   // finalize it
		   (this->*_m_final_vert_hook)(_m_vert_index[2], _m_vert_list[2]);
	   }
   }
   if (edgeTable[cubeindex] & 8) {
	   if (downMost(gridIndex) && (_m_on_which[3] != Vert1 || backMost(gridIndex))) {
		   // finalize it
		   (this->*_m_final_vert_hook)(_m_vert_index[3], _m_vert_list[3]);
	   }
   }
   if (edgeTable[cubeindex] & 16) {
	   if (_m_on_which[4] != Vert2 || rightMost(gridIndex)) {
		   // finalize it
		   (this->*_m_final_vert_hook)(_m_vert_index[4], _m_vert_list[4]);
	   }
   }
   if (edgeTable[cubeindex] & 32) {
	   if (rightMost(gridIndex) && (_m_on_which[5] != Vert2 || backMost(gridIndex))) {
		   // finalize it
		   (this->*_m_final_vert_hook)(_m_vert_index[5], _m_vert_list[5]);
	   }
   }
   if (edgeTable[cubeindex] & 64) {
	   if (backMost(gridIndex) && (_m_on_which[6] != Vert1 || rightMost(gridIndex))) {
		   // finalize it
		   (this->*_m_final_vert_hook)(_m_vert_index[6], _m_vert_list[6]);
	   }
   }
   if (edgeTable[cubeindex] & 128) {
	   if (_m_on_which[7] != Vert1 || backMost(gridIndex)) {
		   // finalize it
		   (this->*_m_final_vert_hook)(_m_vert_index[7], _m_vert_list[7]);
	   }
   }
   if (edgeTable[cubeindex] & 256) {
	   if (_m_on_which[8] != Vert1 || downMost(gridIndex)) {
		   // finalize it
		   (this->*_m_final_vert_hook)(_m_vert_index[8], _m_vert_list[8]);
	   }
   }
   if (edgeTable[cubeindex] & 512) {
	   if (rightMost(gridIndex) && (_m_on_which[9] != Vert1 || downMost(gridIndex))) {
		   // finalize it
		   (this->*_m_final_vert_hook)(_m_vert_index[9], _m_vert_list[9]);
	   }
   }
   if (edgeTable[cubeindex] & 1024) {
	   if (rightBackMost(gridIndex) && (_m_on_which[10] != Vert1 || downMost(gridIndex))) {
		   // finalize it
		   (this->*_m_final_vert_hook)(_m_vert_index[10], _m_vert_list[10]);
	   }
   }
   if (edgeTable[cubeindex] & 2048) {
	   if (backMost(gridIndex) && (_m_on_which[11] != Vert1 || downMost(gridIndex))) {
		   // finalize it1
		   (this->*_m_final_vert_hook)(_m_vert_index[11], _m_vert_list[11]);
	   }
   }

   return n;
}

void MCSimp::genColFinalVertHook(const uint &index, const HVertex &v) {
	CollapsableVertex &cv = _m_pcol->v(index);
	if (cv.finalized())
		return;

	// first erase from the vertex map
	_vertex_map.erase(v);
	_m_pcol->finalizeVert(index);
}

void MCSimp::oocGenFinalVertHook(const uint &index, const HVertex &v) {
	
}

bool MCSimp::genCollapse(
	string filename, double _isovalue, double decimateRate, 
	int *sampleStride, unsigned int maxNewTri, unsigned int &nvert, 
	unsigned int &nface, VolumeSet *paraVolSet) {
	HAugTime htime;
	clearInfo();

	if (decimateRate <= 0 || decimateRate >= 1)
		return false;
	_m_isovalue = _isovalue;
	_m_gen_face_count = 0;
	_m_new_face_count = 0;
	_m_gen_vert_count = 0;

    if (paraVolSet)
        _m_vol_set = *paraVolSet;
    else {
        if (!_m_vol_set.parseDataFile(filename, true, false)) 
            return false;
    }

    if (sampleStride) {
        _m_vol_set.sampleStride.s[0] = sampleStride[0];
        _m_vol_set.sampleStride.s[1] = sampleStride[1];
        _m_vol_set.sampleStride.s[2] = sampleStride[2];
    }

	// set the finalize vertex hook
	_m_final_vert_hook = &MCSimp::genColFinalVertHook;

	if (_m_pcol)
		delete _m_pcol;
	_m_pcol = new QuadricEdgeCollapse();
	UINT4 cubeIndex;
	GRIDCELL cube;
	HFace face[5];

	// init decimation
	if (decimateRate < _m_init_decimate_rate) {
		// first read in maxNewTri triangles and decimate based on initDecimateRate
		while (_m_vol_set.hasNext()) {
			cubeIndex = _m_vol_set.cursor;
			if (!_m_vol_set.nextCube(cube))
				return false;
			polygonise(cubeIndex, cube, face);

			if (_m_new_face_count >= maxNewTri - 2) {
				_m_pcol->targetFace(_m_pcol->collapsableFaces() * _m_init_decimate_rate + _m_pcol->uncollapsableFaces());
				_m_new_face_count = 0;
				break;
			}
		}

		// in each loop read till there are maxNewTri triangles in the buffer
		// the loop stops when the true decimate rate will be lower than the 
		// given decimate rate next time.
		unsigned int initReadCount;
		while (_m_vol_set.hasNext()) {
			while (_m_vol_set.hasNext()) {
				cubeIndex = _m_vol_set.cursor;
				if (!_m_vol_set.nextCube(cube))
					return false;
				polygonise(cubeIndex, cube, face);

				if (_m_pcol->validFaces() >= maxNewTri - 4) {
					// approximated decimate rate of this iteration is
					// lower than the given decimate rate
					if (_m_pcol->collapsableFaces() * _m_init_decimate_rate /
						(_m_gen_face_count - _m_pcol->uncollapsableFaces()) < decimateRate) {
						_m_new_face_count = 0;
						break;
					} else
						_m_pcol->targetFace(_m_pcol->collapsableFaces() * _m_init_decimate_rate + _m_pcol->uncollapsableFaces());
					_m_new_face_count = 0;
					break;
				}
			}
		}
	}

	unsigned int lastCollapsableFaces = _m_pcol->collapsableFaces();
	while (_m_vol_set.hasNext()) {
		cubeIndex = _m_vol_set.cursor;
		if (!_m_vol_set.nextCube(cube))
			return false;
		polygonise(cubeIndex, cube, face);

		if (_m_new_face_count >= maxNewTri - 2) {
			_m_pcol->targetFace(
				lastCollapsableFaces + (_m_pcol->collapsableFaces() - lastCollapsableFaces) * decimateRate 
				+ _m_pcol->uncollapsableFaces());
			lastCollapsableFaces = _m_pcol->collapsableFaces();
			_m_new_face_count = 0;
		}
	}

	_m_pcol->targetFace(_m_gen_face_count * decimateRate);
	
	nvert = _m_pcol->validVerts();
	nface = _m_pcol->validFaces();

	ostringstream oss;
	oss << "#iso surfaces decimated" << endl
        << "#iso value: " << _m_isovalue << endl
		<< "#volume set resolution: " << _m_vol_set.volumeSize.s[0] << "x" << _m_vol_set.volumeSize.s[1] << "x" << _m_vol_set.volumeSize.s[2] << endl 
        << "#sample stride: " << _m_vol_set.sampleStride.s[0] << "x" << _m_vol_set.sampleStride.s[1] << "x" << _m_vol_set.sampleStride.s[2] << endl
        << "#buffer size: " << maxNewTri << endl
		<< "#generated faces: " << _m_gen_face_count << ", vertices: " << _m_gen_vert_count << endl
		<< "#simplified faces: " << nface << ", vertices: " << nvert << endl 
        << "#decimate rate: " << decimateRate << endl
		<< "#time consuming: " << htime.printElapseSec() << endl << endl;
	addInfo(oss.str());

	return true;
}

void MCSimp::toIndexedMesh(HVertex *vertArr, HFace *faceArr) {
	_m_pcol->toIndexedMesh(vertArr, faceArr);
}

void MCSimp::toIndexedMesh(vector<float>& vertArr, vector<int>& faceArr) {
	_m_pcol->toIndexedMesh(vertArr, faceArr);
}

bool MCSimp::oocIndexedGen(string input_file, string output_file, double _isovalue) {
	if (!_m_vol_set.parseDataFile(input_file, true, true)) 
        return false;

	// set the finalize vertex hook
	_m_final_vert_hook = &MCSimp::genColFinalVertHook;

	if (_m_pcol)
		delete _m_pcol;
	_m_pcol = NULL;

	ofstream face_fout;

	_m_gen_vert_count = 0;
	_m_gen_face_count = 0;

	string file_name = getFilename(input_file.c_str());
#ifdef WRITE_FILE_BINARY
	_m_vert_fout.open((file_name+".verts").c_str(), ios::out | ios::binary);
	face_fout.open((file_name+".tris").c_str(), ios::out | ios::binary);
#else
	_m_vert_fout.open((file_name+".verts").c_str(), ios::out | ios::binary);
	face_fout.open((file_name+".tris").c_str(), ios::out | ios::binary);
#endif

	UINT4 cubeIndex;
	GRIDCELL cube;
	int n, fcount;
	HFace face[5];

	while (_m_vol_set.hasNext()) {
		cubeIndex = _m_vol_set.cursor;
		if (!_m_vol_set.nextCube(cube))
			return false;
		n = polygonise(cubeIndex, cube, face);

		for (int i = 0; i < n; i ++) {
			write_face(face_fout, face[i]);
		}
	}

	face_fout.close();
	_m_vert_fout.close();
}