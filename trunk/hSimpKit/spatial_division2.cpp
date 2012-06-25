#include "spatial_division2.h"
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

HVertex HSDVertexCluster2::getRepresentativeVertex()
{
	/* if the qem is not invertible, 
	   it won't modify the input vertex */
	HVertex new_vertex;
	if (awQ.calcRepresentativeVertex(new_vertex))
	{
		// the vertex falls in the cluster
		if (meanVertex != new_vertex &&
			new_vertex.x >= min_x && new_vertex.x <= max_x &&
			new_vertex.y >= min_y && new_vertex.y <= max_y &&
			new_vertex.z >= min_z && new_vertex.z <= max_z) {

			return new_vertex;
		}
	}

	return meanVertex;
}

ostream& operator <<(ostream& out, const HSDVertexCluster2& c)
{
	out << c.getImportance();

	return out;
}

/* -- spatial division class -- */

// threshold of the mean normal treated as a sphere
const float HSpatialDivision2::SPHERE_MEAN_NORMAL_THRESH = 0.2; 
// threshold of the ratio of maximum / minimum curvature treated as a hemisphere
const float HSpatialDivision2::MAX_MIN_CURVATURE_RATIO_TREATED_AS_HEMISPHERE = 2.0;
const float HSpatialDivision2::RANGE_MAX = 1000.0f;

HSpatialDivision2::HSpatialDivision2():
clusters(INIT_HEAP_VOL, MaxHeap)
{
	vertices = NULL;
	vertexCount = 0;
	faces = NULL;
	faceCount = 0;

	fout.open("sddebug.txt");

	vcArrCount = 8;
	vcArr = new HSDVertexCluster2[vcArrCount];
}

HSpatialDivision2::~HSpatialDivision2()
{
	if (vcArr) {
		delete[] vcArr;
	}
}

void HSpatialDivision2::addVertex(HVertex v)
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

void HSpatialDivision2::addFace(HTripleIndex i3)
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
}

bool HSpatialDivision2::readPly(char *filename)
{
	PlyStream plyStream;
	Integer i;
	HVertex v;
	HTripleIndex f;

	if (plyStream.openForRead(filename) == false) {
		return false;
	}

	// set the capacity for the gvl and gfl
	vertices = new HSDVertex2[plyStream.getVertexCount()];
	faces = new HTripleIndex[plyStream.getFaceCount()];
	
	for (i = 0; i < plyStream.getVertexCount(); i ++) {
		if (plyStream.nextVertex(v) == false) {
			return false;
		}
		
		addVertex(v);
	}

	max_range = max_x - min_x;
	if (max_range < max_y - min_y) {
		max_range = max_y - min_y;
	}
	if (max_range < max_z - min_z) {
		max_range = max_z - min_z;
	}

	// resize all coprdinates of vertices to [0, 1000]
	for (i = 0; i < vertexCount; i ++)	{
		vertices[i].x = RANGE_MAX / max_range * (vertices[i].x - min_x);
		vertices[i].y = RANGE_MAX / max_range * (vertices[i].y - min_y);
		vertices[i].z = RANGE_MAX / max_range * (vertices[i].z - min_z);
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

bool HSpatialDivision2::divide(int target_count)
{
	/* - variables - */

	HSDVertexCluster2 vc;
	int i, lastClusterCount = 1, continuousUnchangeCount = 0;
	float diff;

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
	for (i = 0; i < vertexCount; i ++) {
		vc.addVertex(i, vertices[i]);
	}
	for (i = 0; i < faceCount; i ++) {
		vc.addFace(i);
	}
	vc.importance = vc.getImportance();
	clusters.addElement(vc);

	vector<int> index; // index of eigenvalues in eigensolver.eigenvalues()
	index.push_back(0);
	index.push_back(1);
	index.push_back(2);

	// subdivide until the divided clusters reach the target count
	while(clusters.count() < target_count)
	{
		//diff = ((float)target_count - (float)clusters.count()) / target_count;
		//if (diff < 0.01) {
		//	break;
		//}

		if (clusters.count() == lastClusterCount) {
			continuousUnchangeCount ++;
		}
		else {
			continuousUnchangeCount = 0;
		}

		if (continuousUnchangeCount >= 50) {
			cout << "\tstop without reaching the target count because of unchanged cluster count" << endl;
			break;
		}
		
		if (clusters.empty()) {
			cerr << "#error: don't know why but the clusters heap have came to empty" << endl;
			return false;
		}

		// get the value of the top in the heap of clusters and delete it
		lastClusterCount = clusters.count();
		vc = clusters.getTop();
 		clusters.deleteTop();

		//PrintHeap(fout, clusters);

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

			partition8(vc,
				p_nm, HFaceFormula::calcD(p_nm, vc.meanVertex),
				maxDir, HFaceFormula::calcD(maxDir, vc.meanVertex),
				minDir, HFaceFormula::calcD(minDir, vc.meanVertex));
		}
		// partition to 4
		else if (maxCurvature / minCurvature < MAX_MIN_CURVATURE_RATIO_TREATED_AS_HEMISPHERE)
		{
			partition4(vc, 
				maxDir, HFaceFormula::calcD(maxDir, vc.meanVertex),
				minDir, HFaceFormula::calcD(minDir, vc.meanVertex));
		}
		// partition to 2
		else {
			partition2(vc, maxDir, HFaceFormula::calcD(maxDir, vc.meanVertex));
		}

		PrintHeap(fout, clusters);
	}

	//PrintHeap(cout, clusters);

	return true;
}

void HSpatialDivision2::partition8(
	HSDVertexCluster2 vc,
	HNormal n1, float d1, HNormal n2, float d2,
	HNormal n3, float d3) {

	int i;
	list<Integer>::iterator iter;
	HTripleIndex f;

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

	// partition the faces based on that its three vertices belongs to the same subcluster
	for (iter = vc.fIndices->begin(); iter != vc.fIndices->end(); iter ++) {

		f = faces[*iter];

		// if all vertices in a triangle falls in a partitioned cluster
		if (vertices[f.i].clusterIndex == vertices[f.j].clusterIndex &&
			vertices[f.j].clusterIndex == vertices[f.k].clusterIndex) {

			vcArr[vertices[f.i].clusterIndex].addFace(*iter);
		}
	}

	vc.strongClear();

	for (i = 0; i < 8; i ++) {
		splitConnectedRange(vcArr[i]);
	}
}

void HSpatialDivision2::partition4(
	HSDVertexCluster2 vc, 
	HNormal n1, float d1, HNormal n2, float d2) {

	int i;
	list<Integer>::iterator iter;
	HTripleIndex f;

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

	// partition the faces based on that its three vertices belongs to the same subcluster
	for (iter = vc.fIndices->begin(); iter != vc.fIndices->end(); iter ++) {

		f = faces[*iter];

		// if all vertices in a triangle falls in a partitioned cluster
		if (vertices[f.i].clusterIndex == vertices[f.j].clusterIndex &&
			vertices[f.j].clusterIndex == vertices[f.k].clusterIndex) {

			vcArr[vertices[f.i].clusterIndex].addFace(*iter);
		}
	}

	vc.strongClear();

	for (i = 0; i < 4; i ++) {
		splitConnectedRange(vcArr[i]);
	}
}

void HSpatialDivision2::partition2(
	HSDVertexCluster2 vc, 
	HNormal n1, float d1) {

	int i;
	list<Integer>::iterator iter;
	HTripleIndex f;

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

	// partition the faces based on that its three vertices belongs to the same subcluster
	for (iter = vc.fIndices->begin(); iter != vc.fIndices->end(); iter ++) {

		f = faces[*iter];

		// if all vertices in a triangle falls in a partitioned cluster
		if (vertices[f.i].clusterIndex == vertices[f.j].clusterIndex &&
			vertices[f.j].clusterIndex == vertices[f.k].clusterIndex) {

			vcArr[vertices[f.i].clusterIndex].addFace(*iter);
		}
	}

	vc.strongClear();

	for (i = 0; i < 2; i ++) {
		splitConnectedRange(vcArr[i]);
	}
}

void HSpatialDivision2::clear()
{
	if (vertices) {
		delete[] vertices;
	}
	if (faces) {
		delete[] faces;
	}
	clusters.clear();
}

bool HSpatialDivision2::toPly(char *filename)
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
		v.x = v.x * max_range / RANGE_MAX + min_x;
		v.y = v.y * max_range / RANGE_MAX + min_y;
		v.z = v.z * max_range / RANGE_MAX + min_z;
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

void HSpatialDivision2::generateIndexedMesh()
{
	int i, vindex, j, i1, i2, i3;
	HSDVertexCluster2 sdc;
	HTripleIndex tripleIndex;
	int *indexMap = new int[vertexCount];

	for (i = 0; i < clusters.count(); i ++) {
		sdc = clusters.get(i);

		//for (j = sdc.vRangeStart; j <= sdc.vRangeEnd; j ++) {
		//	vertices[j].clusterIndex = i;
		//	indexMap[vertices[j].oldIndex] = j;
		//}
	}

	degFaces.clear();
	for (i = 0; i < faceCount; i ++) {
		i1 = vertices[indexMap[faces[i].i]].clusterIndex;
		i2 = vertices[indexMap[faces[i].j]].clusterIndex;
		i3 = vertices[indexMap[faces[i].k]].clusterIndex;

		if (i1 != i2 && i1 != i3 && i2 != i3) {
			tripleIndex.set(i1, i2, i3);
			degFaces.insert(tripleIndex);
		}
	}

	delete[] indexMap;
}

void HSpatialDivision2::splitConnectedRange(HSDVertexCluster2 &vc)
{
	if (vc.vIndices->size() <= 0)
		return;

	int i;
	list<Integer>::iterator iter;
	HTripleIndex f;
	// local cluster index start from 0, -1 denotes that it hasn't been given a cluster id
	int curCluster = 0;

	// add temporary adjacent faces information to every vertices
	for (iter = vc.fIndices->begin(); iter != vc.fIndices->end(); iter ++) {
		f = faces[*iter];
		vertices[f.i].adjacentFaces.push_back(*iter);
		vertices[f.j].adjacentFaces.push_back(*iter);
		vertices[f.k].adjacentFaces.push_back(*iter);
	}

	// initialize cluster index for all the vertices as -1 (not assigned)
	for (iter = vc.vIndices->begin(); iter != vc.vIndices->end(); iter ++) {
		vertices[*iter].clusterIndex = -1;
	}

	// search and assign the connected clusters the local cluster index
	for (iter = vc.vIndices->begin(); iter != vc.vIndices->end(); iter ++)	{
		// if the vertex hasn't been visited
		if (vertices[*iter].clusterIndex == -1) {
			searchConnectivity(i, curCluster);
			curCluster ++;
		}
	}

	/* -- create vc and add to the heap for every cluster -- */

	if (vcArrCount < curCluster) {
		if (vcArr) {
			delete[] vcArr;
		}

		vcArr = new HSDVertexCluster2[curCluster];
		vcArrCount = curCluster;
	}
	else {
		for (i = 0; i < curCluster; i ++)
			vcArr[i].weakClear();
	}

	// add the vertex indices for every new clusters
	for (iter = vc.vIndices->begin(); iter != vc.vIndices->end(); iter ++) {
		vcArr[vertices[*iter].clusterIndex].addVertex(*iter, vertices[*iter]);
	}

	// add the face indices for every new clusters
	for (iter = vc.fIndices->begin(); iter != vc.fIndices->end(); iter ++) {
		f = faces[*iter];
		if (vertices[f.i].clusterIndex == vertices[f.j].clusterIndex &&
			vertices[f.i].clusterIndex == vertices[f.k].clusterIndex) {
			vcArr[vertices[f.i].clusterIndex].addFace(*iter);
		}
	}

	// clear the temporary adjacent faces information for vertices in the old partitioned cluster
	for (iter = vc.vIndices->begin(); iter != vc.vIndices->end(); iter ++) {
		vertices[*iter].adjacentFaces.clear();
	}

	// add to heap
	for (i = 0; i < curCluster; i ++) {
		if (vcArr[i].hasVertex())
			clusters.addElement(vcArr[i]);
	}
}

void HSpatialDivision2::searchConnectivity(Integer vIndex, Integer clusterIndex) {

	list<Integer>::iterator iter;
	HTripleIndex f;

	vertices[vIndex].clusterIndex = clusterIndex;

	for (iter = vertices[vIndex].adjacentFaces.begin(); iter != vertices[vIndex].adjacentFaces.end(); iter ++) {
		f = faces[*iter];
		// haven't been visited
		if (vertices[f.i].clusterIndex == -1) 
			searchConnectivity(f.i, clusterIndex);
		if (vertices[f.j].clusterIndex == -1) 
			searchConnectivity(f.j, clusterIndex);
		if (vertices[f.k].clusterIndex == -1) 
			searchConnectivity(f.k, clusterIndex);
	}
}