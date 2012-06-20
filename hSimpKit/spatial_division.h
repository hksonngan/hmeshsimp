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

using std::ostream;

#include <Eigen/Eigenvalues>

//#define INIT_HEAP_VOL 10000 //initial heap volume

using std::vector;
using namespace Eigen;

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
	// cluster index
	Integer clusterIndex;
};

class HSDVertexCluster
{
public:
	HSDVertexCluster();
	~HSDVertexCluster() { /*delete[] vIndices;*/ }
	void addVertex(Integer i, HSDVertex v);
	bool operator< (const HSDVertexCluster &vc) const;
	bool operator> (const HSDVertexCluster &vc) const;
	//HSDVertexCluster& operator =(const HSDVertexCluster &vc);
	// clear the object without free the vIndices
	void weakClear();
	// clear the object and free the vIndices
	void strongClear();
	inline float getImportance() const;
	HVertex getRepresentativeVertex();	

public:
	// mean vertex
	HVertex meanVertex;
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
	// bounding box
	float max_x, min_x, max_y, min_y, max_z, min_z;

private:
	//static const int INIT_VERT_VOL = 100;
};

inline float HSDVertexCluster::getImportance() const
{
	HNormal n1(awN.x, awN.y, awN.z);
	float l1 = area - n1.Length();

	return l1;
}

ostream& operator <<(ostream &out, const HSDVertexCluster& c);

/* -- spatial division class -- */
class HSpatialDivision
{
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

public:
	// all the vertices
	vector<HSDVertex> vertices;
	// all the faces
	vector<HTripleIndex> faces;
	// all the clusters in a heap
	doubleHeap<HSDVertexCluster> clusters;
	// degenerated face hash set
	HTripleIndexSet degFaces;

private:
	static const int INIT_HEAP_VOL = 10000; // initial heap capacity
	static const float SPHERE_MEAN_NORMAL_THRESH; // threshold of the mean normal treated as a sphere
	static const float MAX_MIN_CURVATURE_RATIO_TREATED_AS_HEMISPHERE; // threshold of the ratio of maximum / minimum curvature treated as a hemisphere
};

#endif //__SPATIAL_DIVISION__
