/*
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
#include <vector>
#include <boost/unordered_set.hpp>
//#define PRINT_HEAP
#include "double_heap.h"
#include <ostream>
#include <Eigen/Eigenvalues>
#include "algorithm.h"


//#define INIT_HEAP_VOL 10000 //initial heap volume

// constants for the variable vRangeStart to mark 
// if there has been vertex added for computing 
// of bounding box
#define NO_VERTEX -1
#define VERTEX_ADDED -2

#define HDynamArray std::vector

using namespace Eigen;
using std::ostream;


/* -- spatial division vertex -- */

class HSDVertex : public HVertex
{
public:
	HSDVertex() {
		awQ.setZero();
		awN.Set(0.0, 0.0, 0.0);
		area = 0.0;
	}

public:
	// area weighted quadric matrix
	HQEMatrix awQ;
	// area weighted normal
	HNormal awN;
	// area, in fact the area computed is 3 times the de facto area
	float area;
	// cluster or old index before partition
	Integer index;
};

/* the cluster class, mostly a data maintaining class */
class HSDVertexCluster
{
	friend class HSpatialDivision;

public:
	HSDVertexCluster();
	~HSDVertexCluster() { /*delete[] vIndices;*/ }
	void addVertex(Integer i, HSDVertex v);
	bool operator< (const HSDVertexCluster &vc) const;
	bool operator> (const HSDVertexCluster &vc) const;

	/// deprecated
	//HSDVertexCluster& operator =(const HSDVertexCluster &vc);
	///
	
	// clear the object
	void weakClear();
	
	/// deprecated
	// clear the object and free the vIndices
	void strongClear();
	///

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

	/// deprecated
	// vertices indices in the cluster.
	// this pointer will be null unless
	// the function addVertex() is called
	// in case of a waste of memory when
	// value copied, remember to delete
	// the occupied memory space when
	// discarding it
	// vector<Integer> *vIndices;
	///

	// vertex and face range in the gvl, gfl
	Integer vRangeStart, vRangeEnd;
	Integer fRangeStart, fRangeEnd;

	// bounding box
	float max_x, min_x, max_y, min_y, max_z, min_z;

	//static const int INIT_VERT_VOL = 100;
};

inline float HSDVertexCluster::getImportance() const
{
	HNormal n1(awN.x, awN.y, awN.z);
	float l1 = area - n1.Length();

	return l1;
}

ostream& operator <<(ostream &out, const HSDVertexCluster& c);

/* -- spatial division class, mostly a algorithm class -- */
class HSpatialDivision
{
	// constants
private:
	static const int INIT_HEAP_VOL = 10000; // initial heap capacity
	static const float SPHERE_MEAN_NORMAL_THRESH; // threshold of the mean normal treated as a sphere
	static const float MAX_MIN_CURVATURE_RATIO_TREATED_AS_HEMISPHERE; // threshold of the ratio of maximum / minimum curvature treated as a hemisphere
	static const int INIT_V_CAPACITY = 20000; // initial capacity for the vertex container
	static const int INIT_F_CAPACITY = 35000; // initial capacity for the face container

public:
	HSpatialDivision();
	~HSpatialDivision();
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
	void partition8(HSDVertexCluster vc, HSDVertexCluster &vc1,
		HSDVertexCluster &vc2, HSDVertexCluster &vc3,
		HSDVertexCluster &vc4, HSDVertexCluster &vc5,
		HSDVertexCluster &vc6, HSDVertexCluster &vc7,
		HSDVertexCluster &vc8, 
		HNormal n1, float d1, HNormal n2, float d2,
		HNormal n3, float d3);
	void partition4(HSDVertexCluster vc, HSDVertexCluster &vc1,
		HSDVertexCluster &vc2, HSDVertexCluster &vc3,
		HSDVertexCluster &vc4, 
		HNormal n1, float d1, HNormal n2, float d2);
	void partition2(HSDVertexCluster vc, HSDVertexCluster &vc1,
		HSDVertexCluster &vc2, HNormal n1, float d1);
	// check the connectivity of the cluster and add
	// all the connected clusters to heap
	void addUncheckedCluster(HSDVertexCluster &vc);

public:
	// all the vertices, gvl
	HDynamArray<HSDVertex> vertices;
	// all the faces, gfl
	HDynamArray<HTripleIndex> faces;
	// vertex index map
	HDynamArray<Integer> vIndexMap; 
	// all the clusters in a heap
	doubleHeap<HSDVertexCluster> clusters;
	// degenerated face hash set
	HTripleIndexSet degFaces;
	// partition functors
	ArraySelfPartition<HSDVertex, HDynamArray<HSDVertex>> vertPartition;
	ArraySelfPartition<HTripleIndex, HDynamArray<HTripleIndex>> facePartition;
	ElemPartOf<HSDVertex>* vertPartOf[8];
	ElemPartOf<HTripleIndex>* facePartOf[10];
	HFaceFormula faceFormulas[8];
};

/* -- ElemPartOf derivatives -- */

class VertPart1 : public ElemPartOf<HSDVertex>
{
public:
	virtual bool operator() (HSDVertex v);

private:
	HFaceFormula* planes;
	int planeCount;
};

class VertPart2 : public ElemPartOf<HSDVertex>
{
public:
	virtual bool operator() (HSDVertex v);

private:
	HFaceFormula* planes;
	int planeCount;
};

class VertPart3 : public ElemPartOf<HSDVertex>
{
public:
	virtual bool operator() (HSDVertex v);

private:
	HFaceFormula* planes;
	int planeCount;
};

class VertPart4 : public ElemPartOf<HSDVertex>
{
public:
	virtual bool operator() (HSDVertex v);

private:
	HFaceFormula* planes;
	int planeCount;
};

class VertPart5 : public ElemPartOf<HSDVertex>
{
public:
	virtual bool operator() (HSDVertex v);

private:
	HFaceFormula* planes;
	int planeCount;
};

class VertPart6 : public ElemPartOf<HSDVertex>
{
public:
	virtual bool operator() (HSDVertex v);

private:
	HFaceFormula* planes;
	int planeCount;
};

class VertPart7 : public ElemPartOf<HSDVertex>
{
public:
	virtual bool operator() (HSDVertex v);

private:
	HFaceFormula* planes;
	int planeCount;
};

class VertPart8 : public ElemPartOf<HSDVertex>
{
public:
	virtual bool operator() (HSDVertex v);

private:
	HFaceFormula* planes;
	int planeCount;
};

#endif //__SPATIAL_DIVISION__
