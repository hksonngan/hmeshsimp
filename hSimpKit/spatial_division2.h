/*
 *  the vertices and faces division algorithm is based on indices array
 *
 *  INHERITED FROM 'spatial_divison.h':
 *
 *  codes about the algorithm of  'reverse spatial subdivision mesh simplification', 
 *  more detail please refer to 
 *    [Brodsky&Watson] Model Simplification Through Refinement
 *    [Garland] Quadric-based Polygonal Surface Simplification, Chapter 4 Analysis of Quadric Metric
 *
 *	author: ht
 *  email : waytofall916@gmail.com
 */

#ifndef __SPATIAL_DIVISION__
#define __SPATIAL_DIVISION__

#include "util_common.h"
#include "vertex_cluster.h"
#include "hash_face.h"
#include <list>
#include <boost/unordered_set.hpp>
#define PRINT_HEAP
#include "double_heap.h"
#include <ostream>
#include <Eigen/Eigenvalues>
#include "algorithm.h"
#include <fstream>
#include "hdynamarray.h"
//#include "spatial_division.h"

//#define INIT_HEAP_VOL 10000 //initial heap volume

// constants for the variable vRangeStart to mark 
// if there has been vertex added for computing 
// of bounding box
#define NO_VERTEX -2
#define VERTEX_ADDED -3

//#define HDynamArray std::vector

using namespace Eigen;
using std::ostream;
using std::list;
using std::ofstream;

/* class defined */
class HSDVertex2;
class HSDVertexCluster2;
class HSpatialDivision2;

/* -- spatial division vertex -- */

class HSDVertex2 : public HVertex
{
public:
	HSDVertex2() {
		awQ.setZero();
		awN.Set(0.0, 0.0, 0.0);
		area = 0.0;
		clusterIndex = -1;
	}

public:
	// area weighted quadric matrix
	HQEMatrix awQ;
	// area weighted normal
	HNormal awN;
	// area, in fact the area computed is 3 times the de facto area
	float area;
	// cluster index, could be used as local cluster index when checking connectivity
	Integer clusterIndex;
	// adjacent faces, used for checking connectivity
	list<Integer> adjacentFaces;
};

/* the cluster class, mostly a data maintaining class */
class HSDVertexCluster2
{
	friend class HSpatialDivision2;
	friend ostream& operator <<(ostream& out, const HSDVertexCluster2& c);

public:
	static float MINIMUM_NORMAL_VARI; 

public:
	HSDVertexCluster2::HSDVertexCluster2() { weakClear(); }
	~HSDVertexCluster2() { /*delete[] vIndices;*/ }
	inline void addVertex(Integer i, HSDVertex2 v);
	inline void addFace(Integer i);
	
	bool operator <(const HSDVertexCluster2 &vc) const 
		{ return getImportance() < vc.getImportance(); }

	bool operator >(const HSDVertexCluster2 &vc) const 
		{ return getImportance() > vc.getImportance(); }
	
	bool hasVertex()
		{ return vIndices != NULL && vIndices->size() > 0; }

	// clear the object
	inline void weakClear();
	
	// clear the object and free the vIndices
	inline void strongClear();

	inline float getImportance() const;
	HVertex getRepresentativeVertex();

private:
	// mean vertex
	HVertex meanVertex;
	// accumulated area weighted quadric matrix
	HQEMatrix awQ;
	// accumulated area weighted normal
	HNormal awN;
	// accumulated area
	float area;

	float importance;

	// vertices indices in the cluster.
	// this pointer will be null unless
	// the function addVertex() is called
	// in case of a waste of memory when
	// value copied, remember to delete
	// the occupied memory space when
	// discarding it
	list<Integer> *vIndices;
	list<Integer> *fIndices;

	// bounding box
	float max_x, min_x, max_y, min_y, max_z, min_z;

	//static const int INIT_VERT_VOL = 100;
};

inline void HSDVertexCluster2::addVertex(Integer i, HSDVertex2 v)
{
	if (vIndices == NULL || vIndices->size() == 0) {
		max_x = v.x;
		min_x = v.x;
		max_y = v.y;
		min_y = v.y;
		max_z = v.z;
		min_z = v.z;
	}
	else {
		if (v.x > max_x)
			max_x = v.x;
		else if (v.x < min_x) 
			min_x = v.x;

		if (v.y > max_y) 
			max_y = v.y;
		else if (v.y < min_y) 
			min_y = v.y;

		if (v.z > max_z) 
			max_z = v.z;
		else if (v.z < min_z) 
			min_z = v.z;
	}

	if (vIndices == NULL) {
		vIndices = new list<Integer>;
	}

	vIndices->push_back(i);

	meanVertex = meanVertex * (float)(vIndices->size() - 1) / (float)vIndices->size() 
		+ v / (float)vIndices->size();

	this->awN += v.awN;
	this->awQ += v.awQ;
	this->area += v.area;
}

inline void HSDVertexCluster2::addFace(Integer i)
{
	if (fIndices == NULL) {
		fIndices = new list<Integer>;
	}

	fIndices->push_back(i);
}

inline float HSDVertexCluster2::getImportance() const
{
	if (vIndices->size() <= 1) {
		return 0.0;
	}

	HNormal n1(awN.x, awN.y, awN.z);
	float l1 = n1.Length() / area;
	// scale and move the [0, 1] l1 to interval [M, 1]
	l1 = (1.0 - MINIMUM_NORMAL_VARI) * l1 + MINIMUM_NORMAL_VARI;

	return l1 * area;
}

inline void HSDVertexCluster2::weakClear()
{
	awQ.setZero();
	awN.Set(0.0, 0.0, 0.0);
	meanVertex.Set(0.0, 0.0, 0.0);
	vIndices = NULL;
	fIndices = NULL;
	area = 0;
}

inline void HSDVertexCluster2::strongClear()
{
	if (vIndices) {
		delete vIndices;
	}

	if (fIndices) {
		delete fIndices;
	}

	weakClear();
}

ostream& operator <<(ostream &out, const HSDVertexCluster2& c);

/* -- spatial division class, mostly a algorithm class -- */
class HSpatialDivision2
{
	// constants
private:
	static const int INIT_HEAP_VOL = 10000; // initial heap capacity
	static const float SPHERE_MEAN_NORMAL_THRESH; // threshold of the mean normal treated as a sphere
	static const float MAX_MIN_CURVATURE_RATIO_TREATED_AS_HEMISPHERE; // threshold of the ratio of maximum / minimum curvature treated as a hemisphere
	static const int INIT_V_CAPACITY = 20000; // initial capacity for the vertex container
	static const int INIT_F_CAPACITY = 35000; // initial capacity for the face container
	static const float RANGE_MAX;

public:
	HSpatialDivision2();
	~HSpatialDivision2();
	inline void addVertex(HVertex v);
	// caution: better add the faces after 
	// you've added all the vertices
	inline void addFace(HTripleIndex<Integer> i3);
	bool readPly(char *filename);
	bool divide(int target_count);
	bool toPly(char *filename);
	// clear the vertex indices allocated in the heap of HSDVertexCluster
	void clear();
	void generateIndexedMesh();

	int getVertexCount() { return vertexCount; }
	int getFaceCount() { return faceCount; }

private:
	// partition the vertex cluster to 8 4 2 sub clusters 
	// based on the 3 2 1 partition plane
	inline void partition8(
		HSDVertexCluster2 vc,
		HNormal n1, float d1, HNormal n2, float d2,
		HNormal n3, float d3);
	inline void partition4(
		HSDVertexCluster2 vc,
		HNormal n1, float d1, HNormal n2, float d2);
	inline void partition2(
		HSDVertexCluster2 vc, 
		HNormal n1, float d1);

	// split the range of vertices to connected vertex clusters
	inline void splitConnectedRange(HSDVertexCluster2 &vc);
	// recursively depth-first search the connectivity region
	void searchConnectivity(Integer vIndex, Integer clusterIndex);

private:
	// all the vertices, gvl
	HSDVertex2 *vertices; int vertexCount;
	// all the faces, gfl
	HTripleIndex<Integer> *faces; int faceCount;
	// all the clusters in a heap
	doubleHeap<HSDVertexCluster2> clusters;
	// degenerated face hash set
	HTripleIndexSet degFaces;

	// bounding box
	float max_x, min_x, max_y, min_y, max_z, min_z;
	float max_range;

	// some constantly used aiding variables
	WhichSide sideOfPlane1, sideOfPlane2, sideOfPlane3;
	HSDVertexCluster2 vcArr[8];
	HSDVertexCluster2 *vcArr2; int vcArr2Count;

	// debug info
	ofstream fout;
};

inline void HSpatialDivision2::addVertex(HVertex v)
{
	if (vertexCount == 0) {
		max_x = v.x;
		min_x = v.x;
		max_y = v.y;
		min_y = v.y;
		max_z = v.z;
		min_z = v.z;
	}
	else {
		if (v.x > max_x)
			max_x = v.x;
		else if (v.x < min_x)
			min_x = v.x;

		if (v.y > max_y)
			max_y = v.y;
		else if (v.y < min_y)
			min_y = v.y;

		if (v.z > max_z)
			max_z = v.z;
		else if (v.z < min_z)
			min_z = v.z;
	}

	vertices[vertexCount].Set(v.x, v.y, v.z);
	vertexCount ++;
}

inline void HSpatialDivision2::addFace(HTripleIndex<Integer> i3)
{
	faces[faceCount] = i3;
	faceCount ++;

	/* alter the vertices */

	Vec3<float> v1, v2, v3;
	v1.Set(vertices[i3.i].x, vertices[i3.i].y, vertices[i3.i].z);
	v2.Set(vertices[i3.j].x, vertices[i3.j].y, vertices[i3.j].z);
	v3.Set(vertices[i3.k].x, vertices[i3.k].y, vertices[i3.k].z);
	Vec3<float> e1 = v1 - v2;
	Vec3<float> e2 = v2 - v3;
	Vec3<float> nm = e1 ^ e2;
	float area = nm.Length(); // triangle's area is length of cross product of the two edge vectors

	// add area to the corresponding vertices
	vertices[i3.i].area += area;
	vertices[i3.j].area += area;
	vertices[i3.k].area += area;

	nm.Normalize();
	nm *= area;

	// add area weighted normal to the corresponding vertices
	vertices[i3.i].awN += HNormal(nm.x, nm.y, nm.z);
	vertices[i3.j].awN += HNormal(nm.x, nm.y, nm.z);
	vertices[i3.k].awN += HNormal(nm.x, nm.y, nm.z);

	HFaceFormula::calcTriangleFaceFormula(vertices[i3.i], vertices[i3.j], vertices[i3.k]);
	HQEMatrix qem;
	qem.calcQem(HFaceFormula::a, HFaceFormula::b, HFaceFormula::c, HFaceFormula::d);
	qem *= area;

	// add area weighted quadric matrix to the corresponding vertices
	vertices[i3.i].awQ += qem;
	vertices[i3.j].awQ += qem;
	vertices[i3.k].awQ += qem;

	vertices[i3.i].clusterIndex = 0;
	vertices[i3.j].clusterIndex = 0;
	vertices[i3.k].clusterIndex = 0;
}

inline void HSpatialDivision2::partition8(
	HSDVertexCluster2 vc,
	HNormal n1, float d1, HNormal n2, float d2,
	HNormal n3, float d3) {

	int i;
	list<Integer>::iterator iter;
	HTripleIndex<Integer> f;

	for (i = 0; i < 8; i ++) {
		vcArr[i].weakClear();
	}

	// partition the vertices based which side is resides according to the 3 splitting planes
	for (iter = vc.vIndices->begin(); iter != vc.vIndices->end(); iter ++) {

		sideOfPlane1 = HFaceFormula::sideOfPlane(n1, d1, vertices[*iter]);
		sideOfPlane2 = HFaceFormula::sideOfPlane(n2, d2, vertices[*iter]);
		sideOfPlane3 = HFaceFormula::sideOfPlane(n3, d3, vertices[*iter]);

		// the vertex is on which side of the plane
		if (sideOfPlane1 == Side1 && sideOfPlane2 == Side1 && sideOfPlane3 == Side1) {
			vcArr[0].addVertex(*iter, vertices[*iter]);
			vertices[*iter].clusterIndex = 0;
		}
		else if (sideOfPlane1 == Side1 && sideOfPlane2 == Side1 && sideOfPlane3 == Side2) {
			vcArr[1].addVertex(*iter, vertices[*iter]);
			vertices[*iter].clusterIndex = 1;
		}
		else if (sideOfPlane1 == Side1 && sideOfPlane2 == Side2 && sideOfPlane3 == Side1) {
			vcArr[2].addVertex(*iter, vertices[*iter]);
			vertices[*iter].clusterIndex = 2;
		}
		else if (sideOfPlane1 == Side1 && sideOfPlane2 == Side2 && sideOfPlane3 == Side2) {
			vcArr[3].addVertex(*iter, vertices[*iter]);
			vertices[*iter].clusterIndex = 3;
		}
		else if (sideOfPlane1 == Side2 && sideOfPlane2 == Side1 && sideOfPlane3 == Side1) {
			vcArr[4].addVertex(*iter, vertices[*iter]);
			vertices[*iter].clusterIndex = 4;
		}
		else if (sideOfPlane1 == Side2 && sideOfPlane2 == Side1 && sideOfPlane3 == Side2) {
			vcArr[5].addVertex(*iter, vertices[*iter]);
			vertices[*iter].clusterIndex = 5;
		}
		else if (sideOfPlane1 == Side2 && sideOfPlane2 == Side2 && sideOfPlane3 == Side1) {
			vcArr[6].addVertex(*iter, vertices[*iter]);
			vertices[*iter].clusterIndex = 6;
		}
		else {
			vcArr[7].addVertex(*iter, vertices[*iter]);
			vertices[*iter].clusterIndex = 7;
		}
	}

	if (vc.fIndices) {
		// partition the faces based on that its three vertices belongs to the same subcluster
		for (iter = vc.fIndices->begin(); iter != vc.fIndices->end(); iter ++) {

			f = faces[*iter];

			// if all vertices in a triangle falls in a partitioned cluster
			if (vertices[f.i].clusterIndex == vertices[f.j].clusterIndex &&
				vertices[f.j].clusterIndex == vertices[f.k].clusterIndex) {

					vcArr[vertices[f.i].clusterIndex].addFace(*iter);
			}
		}
	}

	vc.strongClear();

	for (i = 0; i < 8; i ++) {
		splitConnectedRange(vcArr[i]);
	}
}

inline void HSpatialDivision2::partition4(
	HSDVertexCluster2 vc, 
	HNormal n1, float d1, HNormal n2, float d2) {

	int i;
	list<Integer>::iterator iter;
	HTripleIndex<Integer> f;

	for (i = 0; i < 4; i ++) {
		vcArr[i].weakClear();
	}

	// partition the vertices based which side is resides according to the 2 splitting planes
	for (iter = vc.vIndices->begin(); iter != vc.vIndices->end(); iter ++) {

		sideOfPlane1 = HFaceFormula::sideOfPlane(n1, d1, vertices[*iter]);
		sideOfPlane2 = HFaceFormula::sideOfPlane(n2, d2, vertices[*iter]);

		// the vertex is on which side of the plane
		if (sideOfPlane1 == Side1 && sideOfPlane2 == Side1) {
			vcArr[0].addVertex(*iter, vertices[*iter]);
			vertices[*iter].clusterIndex = 0;
		}
		else if (sideOfPlane1 == Side1 && sideOfPlane2 == Side2) {
			vcArr[1].addVertex(*iter, vertices[*iter]);
			vertices[*iter].clusterIndex = 1;
		}
		else if (sideOfPlane1 == Side2 && sideOfPlane2 == Side1) {
			vcArr[2].addVertex(*iter, vertices[*iter]);
			vertices[*iter].clusterIndex = 2;
		}
		else {
			vcArr[3].addVertex(*iter, vertices[*iter]);
			vertices[*iter].clusterIndex = 3;
		}
	}

	if (vc.fIndices) {
		// partition the faces based on that its three vertices belongs to the same subcluster
		for (iter = vc.fIndices->begin(); iter != vc.fIndices->end(); iter ++) {

			f = faces[*iter];

			// if all vertices in a triangle falls in a partitioned cluster
			if (vertices[f.i].clusterIndex == vertices[f.j].clusterIndex &&
				vertices[f.j].clusterIndex == vertices[f.k].clusterIndex) {

					vcArr[vertices[f.i].clusterIndex].addFace(*iter);
			}
		}
	}

	vc.strongClear();

	for (i = 0; i < 4; i ++) {
		splitConnectedRange(vcArr[i]);
	}
}

inline void HSpatialDivision2::partition2(
	HSDVertexCluster2 vc, 
	HNormal n1, float d1) {

	int i;
	list<Integer>::iterator iter;
	HTripleIndex<Integer> f;

	for (i = 0; i < 2; i ++) {
		vcArr[i].weakClear();
	}

	// partition the vertices based which side is resides according to the 2 splitting planes
	for (iter = vc.vIndices->begin(); iter != vc.vIndices->end(); iter ++) {

		sideOfPlane1 = HFaceFormula::sideOfPlane(n1, d1, vertices[*iter]);

		// the vertex is on which side of the plane
		if (sideOfPlane1 == Side1) {
			vcArr[0].addVertex(*iter, vertices[*iter]);
			vertices[*iter].clusterIndex = 0;
		}
		else {
			vcArr[1].addVertex(*iter, vertices[*iter]);
			vertices[*iter].clusterIndex = 1;
		}
	}

	if (vc.fIndices) {
		// partition the faces based on that its three vertices belongs to the same subcluster
		for (iter = vc.fIndices->begin(); iter != vc.fIndices->end(); iter ++) {

			f = faces[*iter];

			// if all vertices in a triangle falls in a partitioned cluster
			if (vertices[f.i].clusterIndex == vertices[f.j].clusterIndex &&
				vertices[f.j].clusterIndex == vertices[f.k].clusterIndex) {

					vcArr[vertices[f.i].clusterIndex].addFace(*iter);
			}
		}
	}

	vc.strongClear();

	for (i = 0; i < 2; i ++) {
		splitConnectedRange(vcArr[i]);
	}
}

inline void HSpatialDivision2::splitConnectedRange(HSDVertexCluster2 &vc)
{
	if (vc.vIndices == NULL || vc.vIndices->size() <= 0)
		return;

	int i;
	list<Integer>::iterator iter;
	HTripleIndex<Integer> f;
	// local cluster index start from 0, -1 denotes that it hasn't been given a cluster id
	int curCluster = 0;

	// add temporary adjacent faces information to every vertices
	if (vc.fIndices) {
		for (iter = vc.fIndices->begin(); iter != vc.fIndices->end(); iter ++) {
			f = faces[*iter];
			vertices[f.i].adjacentFaces.push_back(*iter);
			vertices[f.j].adjacentFaces.push_back(*iter);
			vertices[f.k].adjacentFaces.push_back(*iter);
		}
	}

	// initialize cluster index for all the vertices as -1 (not assigned)
	for (iter = vc.vIndices->begin(); iter != vc.vIndices->end(); iter ++) {
		vertices[*iter].clusterIndex = -1;
	}

	// search and assign the connected clusters the local cluster index
	for (iter = vc.vIndices->begin(); iter != vc.vIndices->end(); iter ++)	{
		// if the vertex hasn't been visited
		if (vertices[*iter].clusterIndex == -1) {
			searchConnectivity(*iter, curCluster);
			curCluster ++;
		}
	}

	/* -- create vc and add to the heap for every cluster -- */

	if (vcArr2Count < curCluster) {
		if (vcArr2) {
			delete[] vcArr2;
		}

		vcArr2 = new HSDVertexCluster2[curCluster];
		vcArr2Count = curCluster;
	}
	else {
		for (i = 0; i < curCluster; i ++)
			vcArr2[i].weakClear();
	}

	// add the vertex indices for every new clusters
	for (iter = vc.vIndices->begin(); iter != vc.vIndices->end(); iter ++) {
		vcArr2[vertices[*iter].clusterIndex].addVertex(*iter, vertices[*iter]);
	}

	if (vc.fIndices) {
		// add the face indices for every new clusters
		for (iter = vc.fIndices->begin(); iter != vc.fIndices->end(); iter ++) {
			f = faces[*iter];
			if (vertices[f.i].clusterIndex == vertices[f.j].clusterIndex &&
				vertices[f.i].clusterIndex == vertices[f.k].clusterIndex) {
					vcArr2[vertices[f.i].clusterIndex].addFace(*iter);
			}
		}
	}

	// clear the temporary adjacent faces information for vertices in the old partitioned cluster
	for (iter = vc.vIndices->begin(); iter != vc.vIndices->end(); iter ++) {
		vertices[*iter].adjacentFaces.clear();
	}

	vc.strongClear();

	// add to heap
	for (i = 0; i < curCluster; i ++) {
		if (vcArr2[i].hasVertex())
			clusters.addElement(vcArr2[i]);
	}
}

#endif //__SPATIAL_DIVISION__
