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
#include <vector>
#include "double_heap.h"

//#define INIT_HEAP_VOL 10000 //initial heap volume

using std::vector;

/* spatial division vertex */

typedef HVertex HNormal;

class HSDVertex : public HVertex
{
public:
	HSDVertex()
	{
		awQ.setZero();
		awN.set(0.0, 0.0, 0.0);
		area = 0;
	}

public:
	// area weighted quadric matrix
	HQEMatrix awQ;
	// area weighted normal
	HNormal awN;
	// area
	HNormal area;
};

class HSDVertexCluster
{
public:
	HSDVertexCluster();
	~HSDVertexCluster() { /*delete[] vIndices;*/ }
	void addVertex(Integer i, HQEMatrix q, HNormal awN, float _area);
	bool operator< (const HSDVertexCluster &vc) const;

public:
	// accumulated area weighted quadric matrix
	HQEMatrix awQ;
	// accumulated area weighted normal
	HNormal awN;
	// accumulated area
	float area;
	// vertices indices in the cluster.
	// this pointer will be null unless
	// the function addVertex() is called
	// in case of a waste of memory when
	// value copied, remember to delete
	// the occupied memory space when
	// discarding it
	vector<Integer> *vIndices;

private:
	//static const int INIT_VERT_VOL = 100;
};

/* spatial division class */
class HSpatialDivision
{
public:
	HSpatialDivision();
	~HSpatialDivision();
	void addVertex(HVertex v);
	// caution: better add the faces after 
	// you've added all the vertices
	void addFace(HTripleIndex i3);
	void initFirtCluster(HSDVertexCluster &vc);
	void divide(int target_count);
	void toPly();
	// clear the vertex indices allocated in the heap of HSDVertexCluster
	void clear();

public:
	vector<HSDVertex> vertices;
	vector<HTripleIndex> faces;
	doubleHeap<HSDVertexCluster> clusters;

private:
	static const int INIT_HEAP_VOL = 10000; //initial heap volume
};

#endif //__SPATIAL_DIVISION__
