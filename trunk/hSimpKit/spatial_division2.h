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

public:
	HSDVertexCluster2::HSDVertexCluster2() { weakClear(); }
	~HSDVertexCluster2() { /*delete[] vIndices;*/ }
	inline void addVertex(Integer i, HSDVertex2 v);
	inline void addFace(Integer i);
	bool HSDVertexCluster2::operator< (const HSDVertexCluster2 &vc) const 
		{ return getImportance() < vc.getImportance(); }

	bool HSDVertexCluster2::operator> (const HSDVertexCluster2 &vc) const 
		{ return getImportance() > vc.getImportance(); }

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
	list<HTripleIndex> *fIndices;

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
		fIndices = new list<HTripleIndex>;
	}

	fIndices->push_back(i);
}

inline float HSDVertexCluster2::getImportance() const
{
	if (vIndices->size() <= 1) {
		return 0.0;
	}

	HNormal n1(awN.x, awN.y, awN.z);
	float l1 = area - n1.Length();

	return l1;
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
	void addVertex(HVertex v);
	// caution: better add the faces after 
	// you've added all the vertices
	void addFace(HTripleIndex i3);
	bool readPly(char *filename);
	bool divide(int target_count);
	bool toPly(char *filename);
	// clear the vertex indices allocated in the heap of HSDVertexCluster
	void clear();
	void generateIndexedMesh();

private:
	// partition the vertex cluster to 8 4 2 sub clusters 
	// based on the 3 2 1 partition plane
	void partition8(
		HSDVertexCluster2 vc,
		HNormal n1, float d1, HNormal n2, float d2,
		HNormal n3, float d3);
	void partition4(
		HSDVertexCluster2 vc,
		HNormal n1, float d1, HNormal n2, float d2);
	void partition2(
		HSDVertexCluster2 vc, 
		HNormal n1, float d1);

	// split the range of vertices to connected vertex clusters
	void splitConnectedRange(HSDVertexCluster2 vc);
	// recursively search the connectivity region
	void searchConnectivity(Integer vIndex, Integer clusterIndex);

private:
	// all the vertices, gvl
	HSDVertex2 *vertices; int vertexCount;
	// all the faces, gfl
	HTripleIndex *faces; int faceCount;
	// all the clusters in a heap
	doubleHeap<HSDVertexCluster2> clusters;
	// degenerated face hash set
	HTripleIndexSet degFaces;

	// bounding box
	float max_x, min_x, max_y, min_y, max_z, min_z;
	float max_range;

	// some constantly used aiding variables
	HSDVertexCluster2 vc[8];
	WhichSide sideOfPlane1, sideOfPlane2, sideOfPlane3;
	HSDVertexCluster2 *vc2, vc2Count;

	// debug info
	ofstream fout;
};

#endif //__SPATIAL_DIVISION__
