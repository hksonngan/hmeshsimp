/*
 *	Vertex cluster and some concerning classes
 *  The algorithm presented here please refer to
 *    [Lindstrom] Out-of-Core Simplification of Large Polygonal Models
 *
 *  Author: Ht
 *  Email : waytofall916 at gmail dot com
 *
 *  Copyright (C) Ht. All rights reserved.
 */


#ifndef __VERTEX_CLUSTER__
#define __VERTEX_CLUSTER__

#include <cstddef>
#include <iostream>
#include <boost/unordered_map.hpp>
#include "math/chapill_vec3.h"
#include "math/chapill_vec4.h"
#include "math/mat44.h"
#include "h_math.h"
#include "common_types.h"
#include "hash_def.h"


/* =================== type & constants ================== */

/* representative vertex calculating policy */
typedef unsigned char RepCalcPolicy;
#define QEM_INV		0
#define MEAN_VERTEX 1

//#define PRINT_HASH
const int INIT_BUCKET_SIZE = 1024 * 1024;

/* class defined */
class HVertexCluster;
class HVertexClusterContainer;
class HVertexClusterSimp;

/* =================== class definition ==================== */

/* data structure depicting a vertex cluster */
class HVertexCluster
{
public:
	HVertexCluster() {
		nverts = 0;
		representative_vertex.Set(0.0f, 0.0f, 0.0f);
		qem.setZero();
	}

	HQEMatrix<float>* getQem() {
		return &qem; }

	HQEMatrix<float>& getQemRef() {
		return qem; }

	HVertex& calcRepresentativeVertex(RepCalcPolicy p, float c_max_x, float c_min_x, float c_max_y, 
										float c_min_y, float c_max_z, float c_min_z);

	HVertex& getRepresentativeVertex() {
		return representative_vertex; }

	uint getRepresentativeIndex() {
		return v_index; }

	void addVertex(HVertex &vertex) {
		// mean vertex
		nverts ++;
		representative_vertex = representative_vertex * (((float)nverts - 1) / (float) nverts) + vertex * (1 / (float) nverts);
	}

	void setVIndex(uint n) {
		v_index = n; }

	uint getVIndex() {
		return v_index; }

private:
	/* accumulated quadric error 
	   metrics for the cluster */
	HQEMatrix<float> qem;
	/* vertex index in the simplified mesh */
	uint v_index;
	/* before inverting the qem to calculate the 
	   minimum distance vertex, it stores the mean
	   vertex iteratively */
	HVertex representative_vertex;
	/* vertices count */
	uint nverts;
};

using boost::unordered::unordered_map;
typedef unordered_map<HTriple<uint>, HVertexCluster, HTripleHash, HTripleEqual> HClusterMap;

/* vertex clusters container */
class HVertexClusterContainer
{
public:
	bool create(uint _x_partition, uint _y_partition, uint _z_partition);
	bool clear();

	HVertexCluster* get(const HTriple<uint> &index) const {
		return get(index.i, index.j, index.k); }

	HVertexCluster* get(const uint i, const uint j, const uint k) const {
		return pp_cluster[i * y_partition * z_partition + j * z_partition + k]; }

	bool exist(const HTriple<uint> &index) const {
		return get(index) != NULL; }

	bool exist(const uint i, const uint j, const uint k) const {
		return get(i, j, k) != NULL; }

	bool addFace(HTriple<uint> index, HSoupTriangle tri) {
		addFace(index.i, index.j, index.k, tri);
		return true;
	}

	// add a face qem to the cluster, create a cluster if the it doesn't exist
	// i, j, k is the cluster index
	bool addFace(uint i, uint j, uint k, HSoupTriangle tri);

	// add a vertex to a corresponding cluster for calculating of mean vertex
	void addVertex(HTriple<uint> index, HVertex vertex){
		addVertex(index.i, index.j, index.k, vertex); }

	void addVertex(uint i, uint j, uint k, HVertex vertex);

	void generateIndexForClusters();

	uint getValidClusterCount() {
		return valid_clusters; }

	void calcAllRepresentativeVertices(RepCalcPolicy p);

private:
	HVertexCluster **pp_cluster;
	// partitions along different dimensions
	uint x_partition; uint y_partition; uint z_partition;
	// maximum clusters
	uint cluster_count;
	// valid cluster count
	uint valid_clusters;
};

/* out-of-core vertex clustering algorithm */
class HVertexClusterSimp
{
public:
	HVertexClusterSimp();
	~HVertexClusterSimp();
	bool create(uint _x_partition, uint _y_partition, uint _z_partition, RepCalcPolicy _p);
	void setBoundBox(float _max_x, float _min_x, float _max_y, float _min_y, float _max_z, float _min_z);
	void clear();
	bool addSoupTriangle(HSoupTriangle triangle);
	bool generateIndexedMesh();
	bool writeToPly(char* filename);
	HTriple<uint> retrieveIndex(HVertex v);
	void getClusterRange(HTriple<uint> index, float &_max_x, float &_min_x, float &_max_y, float &_min_y, float &_max_z, float &_min_z);

private:
	/* use hash map to store the degenerated face index */
	HFaceIndexSet face_set;
	/* vertex clusters */
	HVertexClusterContainer vertex_clusters;
	/* partitions in x y z dimension */
	uint x_partition; uint y_partition; uint z_partition;
	/* bound box */
	float max_x, min_x; float max_y, min_y; float max_z, min_z;
	float x_slice; float y_slice; float z_slice;
	/* representative calculating vertex policy */
	RepCalcPolicy rep_calc_policy;
};

#endif