#include "spatial_division.h"
#include <Eigen/Eigenvalues>
#include <iostream>
#include <vector>
#include <algorithm>
#include "util_common.h"
#include "math/vec3.h"
#include "ply_stream.h"
#include <fstream>

using std::cerr;
using std::cout;
using std::endl;
using std::vector;

/* -- spatial division vertex cluster -- */

HSDVertexCluster::HSDVertexCluster() {
	weakClear();
}

void HSDVertexCluster::addVertex(Integer i, HSDVertex v)
{
	if (vIndices == NULL) {
		vIndices = new vector<Integer>();
	}

	if (vIndices->size() == 0) {
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

	vIndices->push_back(i);

	meanVertex = meanVertex * (float)(vIndices->size() - 1) / (float)vIndices->size()
		+ v / (float)vIndices->size();

	this->awN += v.awN;
	this->awQ += v.awQ;
	this->area += v.area;
}

bool HSDVertexCluster::operator< (const HSDVertexCluster &vc) const
{
	return areaWeightedNormal() < vc.areaWeightedNormal();
}

bool HSDVertexCluster::operator> (const HSDVertexCluster &vc) const
{
	return areaWeightedNormal() > vc.areaWeightedNormal();
}

HVertex HSDVertexCluster::getRepresentativeVertex()
{
	/* if the qem is not invertible, 
	   it won't modify the input vertex */
	HVertex new_vertex;
	awQ.calcRepresentativeVertex(new_vertex);

	// the vertex falls in the cluster
	//if (meanVertex != new_vertex &&
	//	new_vertex.x >= min_x && new_vertex.x <= max_x &&
	//	new_vertex.y >= min_y && new_vertex.y <= max_y &&
	//	new_vertex.z >= min_z && new_vertex.z <= max_z) {

	//	return new_vertex;
	//}

	return meanVertex;
}

//HSDVertexCluster& HSDVertexCluster::operator =(const HSDVertexCluster &vc)
//{
//	this->area = vc.area;
//	this->awN = vc.awN;
//	this->awQ = vc.awQ;
//	this->meanVertex = vc.meanVertex;
//	this->vIndices = vc.vIndices;
//
//	return *this;
//}

void HSDVertexCluster::weakClear()
{
	awQ.setZero();
	awN.Set(0.0, 0.0, 0.0);
	meanVertex.Set(0.0, 0.0, 0.0);
	vIndices = NULL;
	area = 0;
}

void HSDVertexCluster::strongClear()
{
	if (vIndices) {
		delete vIndices;
	}

	weakClear();
}

ostream& operator <<(ostream& out, const HSDVertexCluster& c)
{
	out << c.areaWeightedNormal();

	return out;
}

/* -- spatial division class -- */

// threshold of the mean normal treated as a sphere
const float HSpatialDivision::SPHERE_MEAN_NORMAL_THRESH = 0.2; 
// threshold of the ratio of maximum / minimum curvature treated as a hemisphere
const float HSpatialDivision::MAX_MIN_CURVATURE_RATIO_TREATED_AS_HEMISPHERE = 2.0;

HSpatialDivision::HSpatialDivision()
:clusters(INIT_HEAP_VOL, MaxHeap)
{

}

HSpatialDivision::~HSpatialDivision()
{

}

void HSpatialDivision::addVertex(HVertex v)
{
	HSDVertex sdv;
	sdv.Set(v.x, v.y, v.z);

	vertices.push_back(sdv);
}

void HSpatialDivision::addFace(HTripleIndex i3)
{
	faces.push_back(i3);

	//float area = HFaceFormula::calcTriangleFaceArea(vertices[i3.i], vertices[i3.j], vertices[i3.k]);

	Vec3<float> &v1 = vertices[i3.i], &v2 = vertices[i3.j], &v3 = vertices[i3.k];
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
}

bool HSpatialDivision::readPly(char *filename)
{
	PlyStream plyStream;
	Integer i;
	HVertex v;
	HTripleIndex f;

	if (plyStream.openForRead(filename) == false) {
		return false;
	}
	
	for (i = 0; i < plyStream.getVertexCount(); i ++) {
		if (plyStream.nextVertex(v) == false) {
			return false;
		}
		
		addVertex(v);
	}

	for (i = 0; i < plyStream.getFaceCount(); i ++) {
		if (plyStream.nextFace(f) == false)
		{
			return false;
		}
		
		addFace(f);
	}

	return true;
}

static SelfAdjointEigenSolver<Matrix3f> *solver;

static bool cmp(const int &a, const int &b)
{
	return solver->eigenvalues()(a) > solver->eigenvalues()(b);
}

bool HSpatialDivision::divide(int target_count)
{
	/* - variables - */

	HSDVertexCluster vc, vc1, vc2, vc3, vc4, vc5, vc6, vc7, vc8;
	int i;

	// maximum/minimum curvature and the direction
	float maxCurvature; // maximum curvature
	float minCurvature; // minimum curvature
	HNormal maxDir; // maximum curvature direction
	HNormal minDir; // maximum curvature direction

	SelfAdjointEigenSolver<Matrix3f> eigensolver;
	solver = &eigensolver;
	Matrix3f M;

	/* - routines - */

	// init the first cluster
	for (i = 0; i < this->vertices.size(); i ++)
	{
		vc.addVertex(i, vertices[i]);
	}
	clusters.addElement(vc);

	vector<int> index; // index of eigenvalues in eigensolver.eigenvalues()
	index.push_back(0);
	index.push_back(1);
	index.push_back(2);

	// subdivide until the divided clusters reach the target count
	while(clusters.count() < target_count)
	{
		if (clusters.empty()) {
			cerr << "#error: don't know why but the clusters heap have came to empty" << endl;
			return false;
		}

		// get the value of the top in the heap of clusters and delete it
		vc = clusters.getTop();
		clusters.deleteTop();

		PrintHeap(clusters);

		// get the eigenvalue
		M << vc.awQ.a11, vc.awQ.a12, vc.awQ.a13,
			 vc.awQ.a12, vc.awQ.a22, vc.awQ.a23,
			 vc.awQ.a13, vc.awQ.a23, vc.awQ.a33;
		eigensolver.compute(M);
		if (eigensolver.info() != Success) {
			cerr << "#error: eigenvalues computing error" << endl;
			return false;
		}

		// sort the eigenvalues in descending order by the index
		std::sort(index.begin(), index.end(), cmp);

		// get the maximum/minimum curvature and the direction
		maxCurvature = eigensolver.eigenvalues()(index[1]); // maximum curvature
		minCurvature = eigensolver.eigenvalues()(index[2]); // minimum curvature
		maxDir.Set(eigensolver.eigenvectors()(0, index[1]),
			       eigensolver.eigenvectors()(1, index[1]),
			       eigensolver.eigenvectors()(2, index[1])); // maximum curvature direction
		minDir.Set(eigensolver.eigenvectors()(0, index[2]),
			       eigensolver.eigenvectors()(1, index[2]),
			       eigensolver.eigenvectors()(2, index[2])); // maximum curvature direction

		// partition to 8
		if (vc.awN.Length() / vc.area < SPHERE_MEAN_NORMAL_THRESH)
		{
			HNormal p_nm = maxDir ^ minDir;

			partition8(vc, vc1, vc2, vc3, vc4, vc5, vc6, vc7, vc8,
				p_nm, HFaceFormula::calcD(p_nm, vc.meanVertex),
				maxDir, HFaceFormula::calcD(maxDir, vc.meanVertex),
				minDir, HFaceFormula::calcD(minDir, vc.meanVertex));

			if (vc1.vIndices && vc1.vIndices->size() > 0) {
				clusters.addElement(vc1);
				PrintHeap(clusters);
			}
			if (vc2.vIndices && vc2.vIndices->size() > 0) {
				clusters.addElement(vc2);
				PrintHeap(clusters);
			}
			if (vc3.vIndices && vc3.vIndices->size() > 0) {
				clusters.addElement(vc3);
				PrintHeap(clusters);
			}
			if (vc4.vIndices && vc4.vIndices->size() > 0) {
				clusters.addElement(vc4);
				PrintHeap(clusters);
			}
			if (vc5.vIndices && vc5.vIndices->size() > 0) {
				clusters.addElement(vc5);
				PrintHeap(clusters);
			}
			if (vc6.vIndices && vc6.vIndices->size() > 0) {
				clusters.addElement(vc6);
				PrintHeap(clusters);
			}
			if (vc7.vIndices && vc7.vIndices->size() > 0) {
				clusters.addElement(vc7);
				PrintHeap(clusters);
			}
			if (vc8.vIndices && vc8.vIndices->size() > 0) {
				clusters.addElement(vc8);
				PrintHeap(clusters);
			}

			// free space of vc
			vc.strongClear();
			vc1.weakClear();
			vc2.weakClear();
			vc3.weakClear();
			vc4.weakClear();
			vc5.weakClear();
			vc6.weakClear();
			vc7.weakClear();
			vc8.weakClear();
		}
		// partition to 4
		else if (maxCurvature / minCurvature < MAX_MIN_CURVATURE_RATIO_TREATED_AS_HEMISPHERE)
		{
			partition4(vc, vc1, vc2, vc3, vc4,
				maxDir, HFaceFormula::calcD(maxDir, vc.meanVertex),
				minDir, HFaceFormula::calcD(minDir, vc.meanVertex));

			if (vc1.vIndices && vc1.vIndices->size() > 0) {
				clusters.addElement(vc1);
				PrintHeap(clusters);
			}
			if (vc2.vIndices && vc2.vIndices->size() > 0) {
				clusters.addElement(vc2);
				PrintHeap(clusters);
			}
			if (vc3.vIndices && vc3.vIndices->size() > 0) {
				clusters.addElement(vc3);
				PrintHeap(clusters);
			}
			if (vc4.vIndices && vc4.vIndices->size() > 0) {
				clusters.addElement(vc4);
				PrintHeap(clusters);
			}

			// free space of vc
			vc.strongClear();
			vc1.weakClear();
			vc2.weakClear();
			vc3.weakClear();
			vc4.weakClear();
		}
		// partition to 2
		else
		{
			partition2(vc, vc1, vc2, maxDir, HFaceFormula::calcD(maxDir, vc.meanVertex));

			if (vc1.vIndices && vc1.vIndices->size() > 0) {
				clusters.addElement(vc1);
				PrintHeap(clusters);
			}
			if (vc2.vIndices && vc2.vIndices->size() > 0) {
				clusters.addElement(vc2);
				PrintHeap(clusters);
			}

			// free space of vc
			vc.strongClear();
			vc1.weakClear();
			vc2.weakClear();
		}
	}

	return true;
}

inline void HSpatialDivision::partition8(HSDVertexCluster vc, HSDVertexCluster &vc1,
				HSDVertexCluster &vc2, HSDVertexCluster &vc3,
				HSDVertexCluster &vc4, HSDVertexCluster &vc5,
				HSDVertexCluster &vc6, HSDVertexCluster &vc7,
				HSDVertexCluster &vc8, 
				HNormal n1, float d1, HNormal n2, float d2,
				HNormal n3, float d3) {

	if (vc.vIndices == NULL) {
		return;
	}

	int i, j;

	for (i = 0; i < vc.vIndices->size(); i ++)
	{
		j = vc.vIndices->at(i);

		WhichSide sideOfPlane1 = HFaceFormula::sideOfPlane(n1, d1, this->vertices[j]);
		WhichSide sideOfPlane2 = HFaceFormula::sideOfPlane(n2, d2, this->vertices[j]);
		WhichSide sideOfPlane3 = HFaceFormula::sideOfPlane(n3, d3, this->vertices[j]);

		if      (sideOfPlane1 == Side1 && sideOfPlane2 == Side1 && sideOfPlane3 == Side1) {
			vc1.addVertex(j, this->vertices[j]);
		}
		else if (sideOfPlane1 == Side1 && sideOfPlane2 == Side1 && sideOfPlane3 == Side2) {
			vc2.addVertex(j, this->vertices[j]);
		}
		else if (sideOfPlane1 == Side1 && sideOfPlane2 == Side2 && sideOfPlane3 == Side1) {
			vc3.addVertex(j, this->vertices[j]);
		}
		else if (sideOfPlane1 == Side1 && sideOfPlane2 == Side2 && sideOfPlane3 == Side2) {
			vc4.addVertex(j, this->vertices[j]);
		}
		else if (sideOfPlane1 == Side2 && sideOfPlane2 == Side1 && sideOfPlane3 == Side1) {
			vc5.addVertex(j, this->vertices[j]);
		}
		else if (sideOfPlane1 == Side2 && sideOfPlane2 == Side1 && sideOfPlane3 == Side2) {
			vc6.addVertex(j, this->vertices[j]);
		}
		else if (sideOfPlane1 == Side2 && sideOfPlane2 == Side2 && sideOfPlane3 == Side1) {
			vc7.addVertex(j, this->vertices[j]);
		}
		else {
			vc8.addVertex(j, this->vertices[j]);
		}
	}
}

inline void HSpatialDivision::partition4(HSDVertexCluster vc, HSDVertexCluster &vc1,
				HSDVertexCluster &vc2, HSDVertexCluster &vc3,
				HSDVertexCluster &vc4, 
				HNormal n1, float d1, HNormal n2, float d2) {

	if (vc.vIndices == NULL) {
		return;
	}

	int i, j;

	for (i = 0; i < vc.vIndices->size(); i ++)
	{
		j = vc.vIndices->at(i);
		WhichSide sideOfPlane1 = HFaceFormula::sideOfPlane(n1, d1, vertices[j]);
		WhichSide sideOfPlane2 = HFaceFormula::sideOfPlane(n2, d2, vertices[j]);

		if      (sideOfPlane1 == Side1 && sideOfPlane2 == Side1) {
			vc1.addVertex(j, this->vertices[j]);
		}
		else if (sideOfPlane1 == Side1 && sideOfPlane2 == Side2) {
			vc2.addVertex(j, this->vertices[j]);
		}
		else if (sideOfPlane1 == Side2 && sideOfPlane2 == Side1) {
			vc3.addVertex(j, this->vertices[j]);
		}
		else {
			vc4.addVertex(j, this->vertices[j]);
		}
	}
}

inline void HSpatialDivision::partition2(HSDVertexCluster vc, HSDVertexCluster &vc1,
				HSDVertexCluster &vc2, HNormal n1, float d1) {

	if (vc.vIndices == NULL) {
		return;
	}

	int i, j;

	for (i = 0; i < vc.vIndices->size(); i ++)
	{
		j = vc.vIndices->at(i);
		WhichSide sideOfPlane1 = HFaceFormula::sideOfPlane(n1, d1, vertices[j]);	

		if (sideOfPlane1 == Side1) {
			vc1.addVertex(j, this->vertices[j]);
		}
		else {
			vc2.addVertex(j, this->vertices[j]);
		}
	}
}

void HSpatialDivision::clear()
{
	vertices.clear();
	faces.clear();
	for (int i = 0; i < clusters.count(); i ++)
	{
		delete clusters.get(i).vIndices;
	}
	clusters.clear();
}

bool HSpatialDivision::toPly(char *filename)
{
	std::ofstream fout(filename);
	if (fout.bad()) {
		return false;
	}

	generateIndexedMesh();

	/* write head */
	fout << "ply" << endl;
	fout << "format ascii 1.0" << endl;
	fout << "comment generated by ht spatial division simplification" << endl;

	fout << "element vertex " << clusters.count() << endl;
	fout << "property float x" << endl;
	fout << "property float y" << endl;
	fout << "property float z" << endl;
	fout << "element face " << degFaces.size() << std::endl;
	fout << "property list uchar int vertex_indices" << std::endl;
	fout << "end_header" << endl;

	int i;
	HVertex v;

	for (i = 0; i < clusters.count(); i ++) {
		v = clusters.get(i).getRepresentativeVertex();
		fout << v.x << " " << v.y << " " << v.z << endl;
	}

	HTripleIndexSet::iterator iter;
	for (iter = degFaces.begin(); iter != degFaces.end(); iter ++)
	{
		fout << "3 " << iter->i << " " << iter->j << " " << iter->k << std::endl;
	}

	// statistics
	cout << "\twrite simplified mesh successfully" << endl
		<< "\tfile name: " << filename << endl
		<< "\tvertex count: " << clusters.count() << endl
		<< "\tface count: " <<  degFaces.size() << endl;

	return true;
}

void HSpatialDivision::generateIndexedMesh()
{
	int i, vindex, j, i1, i2, i3;
	HSDVertexCluster sdc;
	HTripleIndex tripleIndex;

	for (i = 0; i < clusters.count(); i ++) {
		sdc = clusters.get(i);

		for (j = 0; j < sdc.vIndices->size(); j ++) {
			vindex = sdc.vIndices->at(j);
			vertices[vindex].clusterIndex = i;
		}
	}

	degFaces.clear();
	for (i = 0; i < faces.size(); i ++) {
		i1 = vertices[faces[i].i].clusterIndex;
		i2 = vertices[faces[i].j].clusterIndex;
		i3 = vertices[faces[i].k].clusterIndex;

		if (i1 != i2 && i1 != i3 && i2 != i3) {
			tripleIndex.set(i1, i2, i3);
			degFaces.insert(tripleIndex);
		}
	}
}