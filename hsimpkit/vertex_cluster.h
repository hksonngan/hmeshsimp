/*
 *	Vertex cluster and some concerning classes
 *  The algorithm presented here please refer to
 *    [Lindstrom] Out-of-Core Simplification of Large Polygonal Models
 *
 *  Author: Ht
 *  Email : waytofall916@gmail.com
 *
 *  Copyright (C) Ht-waytofall. All rights reserved.
 *	
 *  This file is part of hmeshsimp.
 *
 *  hmeshsimp is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  hmeshsimp is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with hmeshsimp.  If not, see <http://www.gnu.org/licenses/>.
 */


#ifndef __VERTEX_CLUSTER__
#define __VERTEX_CLUSTER__

//#include <hash_set>
//#include <unordered_set>
#include <cstddef>
#include <iostream>
#include "math/chapill_vec3.h"
#include "math/chapill_vec4.h"
#include "math/mat44.h"
#include "util_common.h"
#include "hash_face.h"


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

	Integer getRepresentativeIndex() {
		return v_index; }

	void addVertex(HVertex &vertex) {
		// mean vertex
		nverts ++;
		representative_vertex = representative_vertex * (((float)nverts - 1) / (float) nverts) + vertex * (1 / (float) nverts);
	}

	void setVIndex(Integer n) {
		v_index = n; }

	Integer getVIndex() {
		return v_index; }

private:
	/* accumulated quadric error 
	   metrics for the cluster */
	HQEMatrix<float> qem;
	/* vertex index in the simplified mesh */
	Integer v_index;
	/* before inverting the qem to calculate the 
	   minimum distance vertex, it stores the mean
	   vertex iteratively
	*/
	HVertex representative_vertex;
	/* vertices count */
	Integer nverts;
};

/* vertex clusters container */
class HVertexClusterContainer
{
public:
	bool create(Integer _x_partition, Integer _y_partition, Integer _z_partition);
	bool clear();

	HVertexCluster* get(const HTripleIndex<Integer> &index) {
		return get(index.i, index.j, index.k); }

	HVertexCluster* get(Integer i, Integer j, Integer k) {
		return pp_cluster[i * y_partition * z_partition + j * z_partition + k]; }

	bool exist(const HTripleIndex<Integer> &index) {
		return get(index) != NULL; }

	bool exist(Integer i, Integer j, Integer k) {
		return get(i, j, k) != NULL; }

	bool addFace(HTripleIndex<Integer> index, HSoupTriangle tri) {
		addFace(index.i, index.j, index.k, tri);
		return true;
	}

	// add a face qem to the cluster, create a cluster if the it doesn't exist
	bool addFace(Integer i, Integer j, Integer k, HSoupTriangle tri);

	// add a vertex to a corresponding cluster for calculating of mean vertex
	void addVertex(HTripleIndex<Integer> index, HVertex vertex){
		addVertex(index.i, index.j, index.k, vertex); }

	void addVertex(Integer i, Integer j, Integer k, HVertex vertex);

	void generateIndexForClusters();

	Integer getValidClusterCount() {
		return valid_clusters; }

	void calcAllRepresentativeVertices(RepCalcPolicy p);

private:
	HVertexCluster **pp_cluster;
	// partitions along different dimensions
	Integer x_partition; Integer y_partition; Integer z_partition;
	// maximum clusters
	Integer cluster_count;
	// valid cluster count
	Integer valid_clusters;
};

/* out-of-core vertex clustering algorithm */
class HVertexClusterSimp
{
public:
	HVertexClusterSimp();
	~HVertexClusterSimp();
	bool create(Integer _x_partition, Integer _y_partition, Integer _z_partition, RepCalcPolicy _p);
	void setBoundBox(float _max_x, float _min_x, float _max_y, float _min_y, float _max_z, float _min_z);
	void clear();
	bool addSoupTriangle(HSoupTriangle triangle);
	bool generateIndexedMesh();
	bool writeToPly(char* filename);
	HTripleIndex<Integer> retrieveIndex(HVertex v);
	void getClusterRange(HTripleIndex<Integer> index, float &_max_x, float &_min_x, float &_max_y, float &_min_y, float &_max_z, float &_min_z);

private:
	/* use hash map to store the degenerated face index */
	HFaceIndexSet face_set;
	/* vertex clusters */
	HVertexClusterContainer vertex_clusters;
	/* partitions in x y z dimension */
	Integer x_partition; Integer y_partition; Integer z_partition;
	/* bound box */
	float max_x, min_x; float max_y, min_y; float max_z, min_z;
	float x_slice; float y_slice; float z_slice;
	/* representative calculating vertex policy */
	RepCalcPolicy rep_calc_policy;
};

#endif