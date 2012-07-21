#include "spatial_division.h"
#include <Eigen/Eigenvalues>
#include <iostream>
#include <vector>
#include <algorithm>
#include "common_types.h"
#include "math/chapill_vec3.h"
#include "ply_stream.h"
#include <fstream>

using std::cerr;
using std::cout;
using std::endl;
using std::vector;

/* -- vertex -- */

void HSDVertex::addConnectivity(uint i) {
	list<uint>::iterator iter;

	// check if i has already existed
	for (iter = connectedVerts.begin(); iter != connectedVerts.end(); iter ++) 
		if ((*iter) == i)
			return;
	
	connectedVerts.push_back(i);
}

/* -- spatial division vertex cluster -- */

HSDVertexCluster::HSDVertexCluster() {
	weakClear();
}

void HSDVertexCluster::addVertex(HSDVertex v)
{
	if (vRangeStart == NO_VERTEX) {
		max_x = v.x;
		min_x = v.x;
		max_y = v.y;
		min_y = v.y;
		max_z = v.z;
		min_z = v.z;
		vRangeStart = VERTEX_ADDED;
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

	vCount ++;
	meanVertex = meanVertex * (float)(vCount - 1) / (float)vCount + v / (float)vCount;

	this->awN += v.awN;
	this->awQ += v.awQ;
	this->area += v.area;
}

bool HSDVertexCluster::operator< (const HSDVertexCluster &vc) const
{
	return getImportance() < vc.getImportance();
}

bool HSDVertexCluster::operator> (const HSDVertexCluster &vc) const
{
	return getImportance() > vc.getImportance();
}

HVertex HSDVertexCluster::getRepresentativeVertex()
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
	//vIndices = NULL;
	vRangeStart = vRangeEnd = NO_VERTEX;
	vCount = 0;
	area = 0;
}

void HSDVertexCluster::strongClear()
{
	//if (vIndices) {
	//	delete vIndices;
	//}

	weakClear();
}

ostream& operator <<(ostream& out, const HSDVertexCluster& c)
{
	out << c.getImportance();

	return out;
}

/* -- spatial division class -- */

// threshold of the mean normal treated as a sphere
const float HSpatialDivision::SPHERE_MEAN_NORMAL_THRESH = 0.2; 
// threshold of the ratio of maximum / minimum curvature treated as a hemisphere
const float HSpatialDivision::MAX_MIN_CURVATURE_RATIO_TREATED_AS_HEMISPHERE = 2.0;
const float HSpatialDivision::RANGE_MAX = 1000.0f;

HSpatialDivision::HSpatialDivision():
clusters(INIT_HEAP_VOL, MaxHeap)
{
	vertPartOf[0] = &vertPart1;
	vertPartOf[1] = &vertPart2;
	vertPartOf[2] = &vertPart3;
	vertPartOf[3] = &vertPart4;
	vertPartOf[4] = &vertPart5;
	vertPartOf[5] = &vertPart6;
	vertPartOf[6] = &vertPart7;
	vertPartOf[7] = &vertPart8;

	vertices = NULL;
	vertexCount = 0;
	faces = NULL;
	faceCount = 0;

	//notifyVertSwap.vertices = &vertices;

	fout.open("sddebug.txt");
}

HSpatialDivision::~HSpatialDivision()
{

}

void HSpatialDivision::addVertex(HVertex v)
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
	vertices[vertexCount].oldIndex = vertexCount;
	vertexCount ++;
}

void HSpatialDivision::addFace(HTriple<uint> i3)
{
	faces[faceCount] = i3;
	faceCount ++;

	/* alter the vertices */

	ChapillVec3<float> v1, v2, v3;
	v1.Set(vertices[i3.i].x, vertices[i3.i].y, vertices[i3.i].z);
	v2.Set(vertices[i3.j].x, vertices[i3.j].y, vertices[i3.j].z);
	v3.Set(vertices[i3.k].x, vertices[i3.k].y, vertices[i3.k].z);
	ChapillVec3<float> e1 = v1 - v2;
	ChapillVec3<float> e2 = v2 - v3;
	ChapillVec3<float> nm = e1 ^ e2;
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
	HQEMatrix<float> qem;
	qem.calcQem(HFaceFormula::a, HFaceFormula::b, HFaceFormula::c, HFaceFormula::d);
	qem *= area;

	// add area weighted quadric matrix to the corresponding vertices
	vertices[i3.i].awQ += qem;
	vertices[i3.j].awQ += qem;
	vertices[i3.k].awQ += qem;

	// add the connectivity info
	vertices[i3.i].addConnectivity(i3.j);
	vertices[i3.i].addConnectivity(i3.k);
	vertices[i3.j].addConnectivity(i3.i);
	vertices[i3.j].addConnectivity(i3.k);
	vertices[i3.k].addConnectivity(i3.i);
	vertices[i3.k].addConnectivity(i3.j);
}

bool HSpatialDivision::readPly(char *filename)
{
	PlyStream plyStream;
	uint i;
	HVertex v;
	HTriple<uint> f;

	if (plyStream.openForRead(filename) == false) {
		return false;
	}

	// set the capacity for the gvl and gfl
	vertices = new HSDVertex[plyStream.getVertexCount()];
	faces = new HTriple<uint>[plyStream.getFaceCount()];
	//vIndexMap.resize(plyStream.getVertexCount());
	notifyVertSwap.vertices = vertices;
	
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

bool HSpatialDivision::divide(int target_count)
{
	/* - variables - */

	HSDVertexCluster vc;
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
	vc.vRangeStart = 0;
	vc.vRangeEnd = vertexCount - 1;
	for (i = 0; i < vertexCount; i ++) {
		vc.addVertex(vertices[i]);
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
		diff = ((float)target_count - (float)clusters.count()) / target_count;
		if (diff < 0.01) {
			break;
		}

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

void HSpatialDivision::partition8(
				HSDVertexCluster vc,
				HNormal n1, float d1, HNormal n2, float d2,
				HNormal n3, float d3) {

	if (vc.vCount == 0) {
		return;
	}

	planes[0].set(n1, d1);
	planes[1].set(n2, d2);
	planes[2].set(n3, d3);

	vertPart1.planeCount = 3;
	vertPart1.planes = planes;

	vertPart2.planeCount = 3;
	vertPart2.planes = planes;

	vertPart3.planeCount = 3;
	vertPart3.planes = planes;

	vertPart4.planeCount = 3;
	vertPart4.planes = planes;

	vertPart5.planeCount = 3;
	vertPart5.planes = planes;

	vertPart6.planeCount = 3;
	vertPart6.planes = planes;

	vertPart7.planeCount = 3;
	vertPart7.planes = planes;

	vertPart8.planeCount = 3;
	vertPart8.planes = planes;

	int* partitionIndices = vertPartition(vertices, vc.vRangeStart, vc.vRangeEnd, vertPartOf, 8, &notifyVertSwap);
	int i, vcStart, vcEnd;

	for (i = 0, vcStart = vc.vRangeStart; i < 8; i ++) {
		vcEnd = partitionIndices[i];
		splitConnectedRange(vcStart, vcEnd);
		vcStart = vcEnd + 1;
	}
}

void HSpatialDivision::partition4(
				HSDVertexCluster vc, 
				HNormal n1, float d1, HNormal n2, float d2) {

	if (vc.vCount == 0) {
		return;
	}

	planes[0].set(n1, d1);
	planes[1].set(n2, d2);

	vertPart1.planeCount = 2;
	vertPart1.planes = planes;

	vertPart2.planeCount = 2;
	vertPart2.planes = planes;

	vertPart3.planeCount = 2;
	vertPart3.planes = planes;

	vertPart4.planeCount = 2;
	vertPart4.planes = planes;

	int* partitionIndices = vertPartition(vertices, vc.vRangeStart, vc.vRangeEnd, vertPartOf, 4, &notifyVertSwap);
	int i, vcStart, vcEnd;

	for (i = 0, vcStart = vc.vRangeStart; i < 4; i ++) {
		vcEnd = partitionIndices[i];
		splitConnectedRange(vcStart, vcEnd);
		vcStart = vcEnd + 1;
	}
}

void HSpatialDivision::partition2(
				HSDVertexCluster vc, 
				HNormal n1, float d1) {

	if (vc.vCount == 0) {
		return;
	}

	planes[0].set(n1, d1);

	vertPart1.planeCount = 1;
	vertPart1.planes = planes;

	vertPart2.planeCount = 1;
	vertPart2.planes = planes;

	int* partitionIndices = vertPartition(vertices, vc.vRangeStart, vc.vRangeEnd, vertPartOf, 2, &notifyVertSwap);
	int i, vcStart, vcEnd;

	for (i = 0, vcStart = vc.vRangeStart; i < 2; i ++) {
		vcEnd = partitionIndices[i];
		splitConnectedRange(vcStart, vcEnd);
		vcStart = vcEnd + 1;
	}
}

void HSpatialDivision::clear()
{
	if (vertices) {
		delete[] vertices;
	}
	if (faces) {
		delete[] faces;
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
		v.x = v.x * max_range / RANGE_MAX + min_x;
		v.y = v.y * max_range / RANGE_MAX + min_y;
		v.z = v.z * max_range / RANGE_MAX + min_z;
		fout << v.x << " " << v.y << " " << v.z << endl;
	}

	HTripleSet::iterator iter;
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
	HTriple<uint> tripleIndex;
	int *indexMap = new int[vertexCount];

	for (i = 0; i < clusters.count(); i ++) {
		sdc = clusters.get(i);

		for (j = sdc.vRangeStart; j <= sdc.vRangeEnd; j ++) {
			vertices[j].clusterIndex = i;
			indexMap[vertices[j].oldIndex] = j;
		}
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

void HSpatialDivision::splitConnectedRange(uint start, uint end)
{
	if (start > end)
		return;

	int i;
	list<uint>::iterator iter;
	// local cluster index start from 1, 0 denotes that it hasn't been given a cluster id
	unsigned short curCluster = 1;

	// erase the adjacent index out of the range
	for (i = start; i <= end; i ++) {
		vertices[i].clusterIndex = 0;
		for (iter = vertices[i].connectedVerts.begin(); iter != vertices[i].connectedVerts.end();)
			if (*iter < start || *iter > end) 
				iter = vertices[i].connectedVerts.erase(iter);
			else
				iter ++;
	}

	// search and assign the connected clusters the local cluster index
	for (i = start; i <= end; i ++)	{
		// if the vertex hasn't been visited
		if (vertices[i].clusterIndex == 0) {
			searchConnectivity(i, start, curCluster);
			curCluster ++;
		}
	}

	// curCluster - 1 is the count of connected clusters
	ElemPartOf<HSDVertex>** partOf = new ElemPartOf<HSDVertex>*[curCluster - 1];
	// retrieve the condition functor for every cluster
	for (i = 1; i < curCluster; i ++) {
		partOf[i - 1] = new VertPartofCluster();
		((VertPartofCluster*)partOf[i - 1])->cIndex = i;
	}

	int *partitionIndices = vertPartition2(vertices, start, end, partOf, curCluster - 1, &notifyVertSwap);

	// create vc and add to the heap for every cluster
	HSDVertexCluster vc;
	int vcStart, vcEnd, j;
	for (i = 0, vcStart = start; i < curCluster - 1; i ++) {
		vcEnd = partitionIndices[i];
		if (vcStart <= vcEnd) {
			vc.weakClear();
			for (j = vcStart; j <= vcEnd; j ++) {
				vc.addVertex(vertices[j]);
			}
			vc.vRangeStart = vcStart;
			vc.vRangeEnd = vcEnd;
			clusters.addElement(vc);
			vc.importance = vc.getImportance();

			//PrintHeap(fout, clusters);
		}
		vcStart = vcEnd + 1;
	}

	for (i = i; i < curCluster; i ++)
		delete partOf[i - 1];
	delete[] partOf;
}

void HSpatialDivision::searchConnectivity(uint vIndex, uint rangeStart, uint clusterIndex) {

	list<uint>::iterator iter;
	vertices[vIndex].clusterIndex = clusterIndex;

	for (iter = vertices[vIndex].connectedVerts.begin(); iter != vertices[vIndex].connectedVerts.end(); iter ++)
		// haven't been visited
		if (*iter != vIndex && vertices[*iter].clusterIndex == 0) 
			searchConnectivity(*iter, rangeStart, clusterIndex);
}

/* -- ElemPartOf drivatives -- */

bool VertPart1::operator() (HSDVertex v) {

	if (planeCount == 1) {

		WhichSide sideOfPlane1 = HFaceFormula::sideOfPlane(planes[0]._a, planes[0]._b, planes[0]._c, planes[0]._d, v);

		if (sideOfPlane1 == Side1) {
			return true;
		}
		else {
			return false;
		}
	}
	else if (planeCount == 2) {

		WhichSide sideOfPlane1 = HFaceFormula::sideOfPlane(planes[0]._a, planes[0]._b, planes[0]._c, planes[0]._d, v);
		WhichSide sideOfPlane2 = HFaceFormula::sideOfPlane(planes[1]._a, planes[1]._b, planes[1]._c, planes[1]._d, v);

		if (sideOfPlane1 == Side1 && sideOfPlane2 == Side1) {
			return true;
		}
		else {
			return false;
		}
	}
	else if (planeCount == 3) {

		WhichSide sideOfPlane1 = HFaceFormula::sideOfPlane(planes[0]._a, planes[0]._b, planes[0]._c, planes[0]._d, v);
		WhichSide sideOfPlane2 = HFaceFormula::sideOfPlane(planes[1]._a, planes[1]._b, planes[1]._c, planes[1]._d, v);
		WhichSide sideOfPlane3 = HFaceFormula::sideOfPlane(planes[2]._a, planes[2]._b, planes[2]._c, planes[2]._d, v);

		if (sideOfPlane1 == Side1 && sideOfPlane2 == Side1 && sideOfPlane3 == Side1) {
			return true;
		}
		else {
			return false;
		}
	}

	return false;
}

bool VertPart2::operator() (HSDVertex v) {

	if (planeCount == 1) {

		WhichSide sideOfPlane1 = HFaceFormula::sideOfPlane(planes[0]._a, planes[0]._b, planes[0]._c, planes[0]._d, v);

		if (sideOfPlane1 == Side2) {
			return true;
		}
		else {
			return false;
		}
	}
	else if (planeCount == 2) {

		WhichSide sideOfPlane1 = HFaceFormula::sideOfPlane(planes[0]._a, planes[0]._b, planes[0]._c, planes[0]._d, v);
		WhichSide sideOfPlane2 = HFaceFormula::sideOfPlane(planes[1]._a, planes[1]._b, planes[1]._c, planes[1]._d, v);

		if (sideOfPlane1 == Side1 && sideOfPlane2 == Side2) {
			return true;
		}
		else {
			return false;
		}
	}
	else if (planeCount == 3) {

		WhichSide sideOfPlane1 = HFaceFormula::sideOfPlane(planes[0]._a, planes[0]._b, planes[0]._c, planes[0]._d, v);
		WhichSide sideOfPlane2 = HFaceFormula::sideOfPlane(planes[1]._a, planes[1]._b, planes[1]._c, planes[1]._d, v);
		WhichSide sideOfPlane3 = HFaceFormula::sideOfPlane(planes[2]._a, planes[2]._b, planes[2]._c, planes[2]._d, v);

		if (sideOfPlane1 == Side1 && sideOfPlane2 == Side1 && sideOfPlane3 == Side2) {
			return true;
		}
		else {
			return false;
		}
	}

	return false;
}

bool VertPart3::operator() (HSDVertex v) {

	if (planeCount == 2) {

		WhichSide sideOfPlane1 = HFaceFormula::sideOfPlane(planes[0]._a, planes[0]._b, planes[0]._c, planes[0]._d, v);
		WhichSide sideOfPlane2 = HFaceFormula::sideOfPlane(planes[1]._a, planes[1]._b, planes[1]._c, planes[1]._d, v);

		if (sideOfPlane1 == Side2 && sideOfPlane2 == Side1) {
			return true;
		}
		else {
			return false;
		}
	}
	else if (planeCount == 3) {

		WhichSide sideOfPlane1 = HFaceFormula::sideOfPlane(planes[0]._a, planes[0]._b, planes[0]._c, planes[0]._d, v);
		WhichSide sideOfPlane2 = HFaceFormula::sideOfPlane(planes[1]._a, planes[1]._b, planes[1]._c, planes[1]._d, v);
		WhichSide sideOfPlane3 = HFaceFormula::sideOfPlane(planes[2]._a, planes[2]._b, planes[2]._c, planes[2]._d, v);

		if (sideOfPlane1 == Side1 && sideOfPlane2 == Side2 && sideOfPlane3 == Side1) {
			return true;
		}
		else {
			return false;
		}
	}

	return false;
}

bool VertPart4::operator() (HSDVertex v) {

	if (planeCount == 2) {

		WhichSide sideOfPlane1 = HFaceFormula::sideOfPlane(planes[0]._a, planes[0]._b, planes[0]._c, planes[0]._d, v);
		WhichSide sideOfPlane2 = HFaceFormula::sideOfPlane(planes[1]._a, planes[1]._b, planes[1]._c, planes[1]._d, v);

		if (sideOfPlane1 == Side2 && sideOfPlane2 == Side2) {
			return true;
		}
		else {
			return false;
		}
	}
	else if (planeCount == 3) {

		WhichSide sideOfPlane1 = HFaceFormula::sideOfPlane(planes[0]._a, planes[0]._b, planes[0]._c, planes[0]._d, v);
		WhichSide sideOfPlane2 = HFaceFormula::sideOfPlane(planes[1]._a, planes[1]._b, planes[1]._c, planes[1]._d, v);
		WhichSide sideOfPlane3 = HFaceFormula::sideOfPlane(planes[2]._a, planes[2]._b, planes[2]._c, planes[2]._d, v);

		if (sideOfPlane1 == Side1 && sideOfPlane2 == Side2 && sideOfPlane3 == Side2) {
			return true;
		}
		else {
			return false;
		}
	}

	return false;
}

bool VertPart5::operator() (HSDVertex v) {

	if (planeCount == 3) {

		WhichSide sideOfPlane1 = HFaceFormula::sideOfPlane(planes[0]._a, planes[0]._b, planes[0]._c, planes[0]._d, v);
		WhichSide sideOfPlane2 = HFaceFormula::sideOfPlane(planes[1]._a, planes[1]._b, planes[1]._c, planes[1]._d, v);
		WhichSide sideOfPlane3 = HFaceFormula::sideOfPlane(planes[2]._a, planes[2]._b, planes[2]._c, planes[2]._d, v);

		if (sideOfPlane1 == Side2 && sideOfPlane2 == Side1 && sideOfPlane3 == Side1) {
			return true;
		}
		else {
			return false;
		}
	}

	return false;
}

bool VertPart6::operator() (HSDVertex v) {

	if (planeCount == 3) {

		WhichSide sideOfPlane1 = HFaceFormula::sideOfPlane(planes[0]._a, planes[0]._b, planes[0]._c, planes[0]._d, v);
		WhichSide sideOfPlane2 = HFaceFormula::sideOfPlane(planes[1]._a, planes[1]._b, planes[1]._c, planes[1]._d, v);
		WhichSide sideOfPlane3 = HFaceFormula::sideOfPlane(planes[2]._a, planes[2]._b, planes[2]._c, planes[2]._d, v);

		if (sideOfPlane1 == Side2 && sideOfPlane2 == Side1 && sideOfPlane3 == Side2) {
			return true;
		}
		else {
			return false;
		}
	}

	return false;
}

bool VertPart7::operator() (HSDVertex v) {

	if (planeCount == 3) {

		WhichSide sideOfPlane1 = HFaceFormula::sideOfPlane(planes[0]._a, planes[0]._b, planes[0]._c, planes[0]._d, v);
		WhichSide sideOfPlane2 = HFaceFormula::sideOfPlane(planes[1]._a, planes[1]._b, planes[1]._c, planes[1]._d, v);
		WhichSide sideOfPlane3 = HFaceFormula::sideOfPlane(planes[2]._a, planes[2]._b, planes[2]._c, planes[2]._d, v);

		if (sideOfPlane1 == Side2 && sideOfPlane2 == Side2 && sideOfPlane3 == Side1) {
			return true;
		}
		else {
			return false;
		}
	}

	return false;
}

bool VertPart8::operator() (HSDVertex v) {

	if (planeCount == 3) {

		WhichSide sideOfPlane1 = HFaceFormula::sideOfPlane(planes[0]._a, planes[0]._b, planes[0]._c, planes[0]._d, v);
		WhichSide sideOfPlane2 = HFaceFormula::sideOfPlane(planes[1]._a, planes[1]._b, planes[1]._c, planes[1]._d, v);
		WhichSide sideOfPlane3 = HFaceFormula::sideOfPlane(planes[2]._a, planes[2]._b, planes[2]._c, planes[2]._d, v);

		if (sideOfPlane1 == Side2 && sideOfPlane2 == Side2 && sideOfPlane3 == Side2) {
			return true;
		}
		else {
			return false;
		}
	}

	return false;
}
 
void NotifyVertSwap::operator() (int i, int j) {

	if (i == j)
		return;

	ChangeConnecIndicesAfterSwap(i, j);
	ChangeConnecIndicesAfterSwap(j, i);
}


bool VertPartofCluster::operator() (HSDVertex v) {
	return v.clusterIndex == cIndex;
}