/*
 *  the vertices and faces division algorithm algorithm is based on array self partition
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
#include <vector>
#include <list>
#include <boost/unordered_set.hpp>
#define PRINT_HEAP
#include "double_heap.h"
#include <ostream>
#include <Eigen/Eigenvalues>
#include "algorithm.h"
#include <fstream>


//#define INIT_HEAP_VOL 10000 //initial heap volume

// constants for the variable vRangeStart to mark 
// if there has been vertex added for computing 
// of bounding box
#define NO_VERTEX -2
#define VERTEX_ADDED -3

#define HDynamArray std::vector

using namespace Eigen;
using std::ostream;
using std::list;
using std::ofstream;

/* class defined */
class HSDVertex;
class HSDVertexCluster;
class HSpatialDivision;
class VertPart1;
class VertPart2;
class VertPart3;
class VertPart4;
class VertPart5;
class VertPart6;
class VertPart7;
class VertPart8;
class NotifyVertSwap;
class VertPartofCluster;

/* -- spatial division vertex -- */

class HSDVertex : public HVertex
{
public:
	HSDVertex() {
		awQ.setZero();
		awN.Set(0.0, 0.0, 0.0);
		area = 0.0;
		clusterIndex = -1;
	}

	void addConnectivity(Integer i);
	inline void changeConnecIndices(Integer orig_i, Integer changed_i);

public:
	// area weighted quadric matrix
	HQEMatrix awQ;
	// area weighted normal
	HNormal awN;
	// area, in fact the area computed is 3 times the de facto area
	float area;
	// cluster index, could be used as local cluster index when checking connectivity
	Integer clusterIndex;
	// connected vertices
	list<Integer> connectedVerts;
	// index before moved
	Integer oldIndex;
};

void HSDVertex::changeConnecIndices(Integer orig_i, Integer changed_i) {

	list<Integer>::iterator iter;

	for (iter = connectedVerts.begin(); iter != connectedVerts.end(); iter ++)
		if (*iter == orig_i) {
			*iter = changed_i;
			return;
		}
}

/* the cluster class, mostly a data maintaining class */
class HSDVertexCluster
{
	friend class HSpatialDivision;

public:
	HSDVertexCluster();
	~HSDVertexCluster() { /*delete[] vIndices;*/ }
	/// deprecated
	//void addVertex(Integer i, HSDVertex v);
	///
	void addVertex(HSDVertex v);
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

	float importance;

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
	// vertex count
	Integer vCount;

	// bounding box
	float max_x, min_x, max_y, min_y, max_z, min_z;

	//static const int INIT_VERT_VOL = 100;
};

inline float HSDVertexCluster::getImportance() const
{
	if (vCount <= 1) {
		return 0.0;
	}

	HNormal n1(awN.x, awN.y, awN.z);
	float l1 = area - n1.Length();

	return l1;
}

ostream& operator <<(ostream &out, const HSDVertexCluster& c);

/* -- ElemPartOf derivatives, used for partition -- */

class VertPart1 : public ElemPartOf<HSDVertex>
{
	friend class HSpatialDivision;

public:
	virtual bool operator() (HSDVertex v);

private:
	HFaceFormula* planes;
	int planeCount;
};

class VertPart2 : public ElemPartOf<HSDVertex>
{
	friend class HSpatialDivision;

public:
	virtual bool operator() (HSDVertex v);

private:
	HFaceFormula* planes;
	int planeCount;
};

class VertPart3 : public ElemPartOf<HSDVertex>
{
	friend class HSpatialDivision;

public:
	virtual bool operator() (HSDVertex v);

private:
	HFaceFormula* planes;
	int planeCount;
	HSDVertexCluster *vc;
};

class VertPart4 : public ElemPartOf<HSDVertex>
{
	friend class HSpatialDivision;

public:
	virtual bool operator() (HSDVertex v);

private:
	HFaceFormula* planes;
	int planeCount;
};

class VertPart5 : public ElemPartOf<HSDVertex>
{
	friend class HSpatialDivision;

public:
	virtual bool operator() (HSDVertex v);

private:
	HFaceFormula* planes;
	int planeCount;
};

class VertPart6 : public ElemPartOf<HSDVertex>
{
	friend class HSpatialDivision;

public:
	virtual bool operator() (HSDVertex v);

private:
	HFaceFormula* planes;
	int planeCount;
};

class VertPart7 : public ElemPartOf<HSDVertex>
{
	friend class HSpatialDivision;

public:
	virtual bool operator() (HSDVertex v);

private:
	HFaceFormula* planes;
	int planeCount;
};

class VertPart8 : public ElemPartOf<HSDVertex>
{
	friend class HSpatialDivision;

public:
	virtual bool operator() (HSDVertex v);

private:
	HFaceFormula* planes;
	int planeCount;
};

class VertPartofCluster : public ElemPartOf<HSDVertex>
{
	friend class HSpatialDivision;

public:
	virtual bool operator() (HSDVertex v);

private:
	Integer cIndex;
};

/*  take such a connectivity as an example:
 *   v | 2 .. 4 .. 7 ..
 *  ---+---------------- 
 *   c | 1    2    8
 *     | 7    6    2
 *     | 4    5
 *
 *  if we swap 2 and 4, the memory looks like:
 *   v | 2 .. 4 .. 7 ..
 *  ---+---------------- 
 *   c | 2    1    8
 *     | 6    7    2
 *     | 5    4
 *
 *  the ChangeConnecIndicesAfterSwap(2, 4) updates
 *  the original 2th element's connectivity information
 *  by notifying all its adjacent vertices that its
 *  index has changed from 2 to 4, so you have to change
 *  the '2' in 7's adjacent vertices to '4'. However, if
 *  the swapped vertices are adjacent themselves, we don't
 *  go their connectivity vertices array but modify the 
 *  index instantly, so after ChangeConnecIndicesAfterSwap(2, 4)
 *  has been invoked, the memory looks like this, * denote
 *  changed indices:
 *   v | 2 .. 4 .. 7 ..
 *  ---+---------------- 
 *   c | 2    1    8
 *     | 6    7    4*
 *     | 5    2*
 */
class NotifyVertSwap : public NotifySwap {
	friend class HSpatialDivision;

public:
	virtual void operator() (Integer i, Integer j);

private:
	/* change the connected indices after the swap
	 * orig_i:		original index
	 * changed_i:	changed index */
	void ChangeConnecIndicesAfterSwap(int orig_i, int changed_i) {
		
		list<Integer>::iterator iter;
		HSDVertex vert = vertices[changed_i];
		
		for (iter = vert.connectedVerts.begin(); iter != vert.connectedVerts.end(); iter ++) {

			if (*iter == changed_i) {
				*iter = orig_i;
				continue;
			}

			vertices[*iter].changeConnecIndices(orig_i, changed_i);
		}
	}

public:
	HSDVertex* vertices;
};

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
	static const float RANGE_MAX;

public:
	HSpatialDivision();
	~HSpatialDivision();
	void addVertex(HVertex v);
	// caution: better add the faces after 
	// you've added all the vertices
	void addFace(HTripleIndex<Integer> i3);
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
		HSDVertexCluster vc,
		HNormal n1, float d1, HNormal n2, float d2,
		HNormal n3, float d3);
	void partition4(
		HSDVertexCluster vc,
		HNormal n1, float d1, HNormal n2, float d2);
	void partition2(
		HSDVertexCluster vc, 
		HNormal n1, float d1);
	/// deprecated
	//void partition8(HSDVertexCluster vc, HSDVertexCluster &vc1,
	//	HSDVertexCluster &vc2, HSDVertexCluster &vc3,
	//	HSDVertexCluster &vc4, HSDVertexCluster &vc5,
	//	HSDVertexCluster &vc6, HSDVertexCluster &vc7,
	//	HSDVertexCluster &vc8, 
	//	HNormal n1, float d1, HNormal n2, float d2,
	//	HNormal n3, float d3);
	//void partition4(HSDVertexCluster vc, HSDVertexCluster &vc1,
	//	HSDVertexCluster &vc2, HSDVertexCluster &vc3,
	//	HSDVertexCluster &vc4, 
	//	HNormal n1, float d1, HNormal n2, float d2);
	//void partition2(HSDVertexCluster vc, HSDVertexCluster &vc1,
	//	HSDVertexCluster &vc2, HNormal n1, float d1);
	///

	// split the range of vertices to connected vertex clusters
	void splitConnectedRange(Integer start, Integer end);
	// recursively search the connectivity region
	void searchConnectivity(Integer vIndex, Integer rangeStart, Integer clusterIndex);

private:
	// all the vertices, gvl
	HSDVertex *vertices;
	int vertexCount;
	// all the faces, gfl
	HTripleIndex<Integer> *faces;
	int faceCount;
	/// deprecated
	// vertex index map
	//HDynamArray<Integer> vIndexMap; 
	///
	// all the clusters in a heap
	doubleHeap<HSDVertexCluster> clusters;
	// degenerated face hash set
	HTripleIndexSet degFaces;
	// partition functors
	// patition8 4 2 and splitConnectedRange should use different object of ArraySelfPartition
	ArraySelfPartition<HSDVertex, HSDVertex*> vertPartition;
	ArraySelfPartition<HSDVertex, HSDVertex*> vertPartition2;
	ElemPartOf<HSDVertex>* vertPartOf[8];
	VertPart1 vertPart1;
	VertPart2 vertPart2;
	VertPart3 vertPart3;
	VertPart4 vertPart4;
	VertPart5 vertPart5;
	VertPart6 vertPart6;
	VertPart7 vertPart7;
	VertPart8 vertPart8;
	HFaceFormula planes[3];
	NotifyVertSwap notifyVertSwap;

	ofstream fout;

	// bounding box
	float max_x, min_x, max_y, min_y, max_z, min_z;
	float max_range;
};

#endif //__SPATIAL_DIVISION__
