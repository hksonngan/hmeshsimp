#include "spatial_division2.h"
#include <vector>
#include <algorithm>
#include <time.h>
#include "util_common.h"
#include "math/vec3.h"
#include "ply_stream.h"
#include "stdio.h"

using std::cerr;
using std::cout;
using std::endl;
using std::vector;


/* -- spatial division vertex cluster -- */

float HSDVertexCluster2::MINIMUM_NORMAL_VARI = 0.5; 

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
	HNormal n1(c.awN.x, c.awN.y, c.awN.z);
	float l1 = n1.Length() / c.area;
	// scale and move the [0, 1] l1 to interval [M, 1]
	l1 = (1.0 - HSDVertexCluster2::MINIMUM_NORMAL_VARI) * l1 + HSDVertexCluster2::MINIMUM_NORMAL_VARI;

	if (c.vIndices->size() <= 1) {
		out << "0.0" << "=" << l1 << "*" << c.area << " (" << c.vIndices->size() << ")";
		return out;
	}

	out << l1 * c.area << "=" << l1 << "*" << c.area << " (" << c.vIndices->size() << ")";

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

	time_t rawtime;
	struct tm * timeinfo;
	time (&rawtime);
	timeinfo = localtime (&rawtime);

#ifdef PRINT_DEBUG_INFO
	fdebug.open("sddebug.txt", fstream::out | fstream::app);
	fdebug << endl << endl << asctime(timeinfo) << endl; 
#endif

	flog.open("sd.log", fstream::out | fstream::app);
	flog << endl << endl << asctime(timeinfo) << endl; 

	vcArr2Count = 0;
	vcArr2 = NULL;
}

HSpatialDivision2::~HSpatialDivision2()
{
	clear();
}

bool HSpatialDivision2::readPly(char *filename)
{
	PlyStream plyStream;
	Integer i;
	HVertex v;
	HTripleIndex<Integer> f;

	htime.setCheckPoint();

	if (plyStream.openForRead(filename) == false) {
		return false;
	}

	// set the capacity for the gvl and gfl
	vertices = new HSDVertex2[plyStream.getVertexCount()];
	faces = new HTripleIndex<Integer>[plyStream.getFaceCount()];

	float float_val = 5.89458;
	
	for (i = 0; i < plyStream.getVertexCount(); i ++) {
		if (i == 26) {
			int k = 0;
			k ++;
		}

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

	// resize all coordinates of vertices to [0, 1000]
	for (i = 0; i < vertexCount; i ++)	{
		vertices[i].x = RANGE_MAX / max_range * (vertices[i].x - min_x);
		vertices[i].y = RANGE_MAX / max_range * (vertices[i].y - min_y);
		vertices[i].z = RANGE_MAX / max_range * (vertices[i].z - min_z);
	}

	for (i = 0; i < plyStream.getFaceCount(); i ++) {
		if (plyStream.nextFace(f) == false) {
			return false;
		}
		
		addFace(f);
	}

	cout << "\t-----------------------------------------------" << endl 
		 << "\tread file successfully" << endl
		 << "\tfile name:\t" << filename << endl
		 << "\tvertex count:\t" << getVertexCount() << "\tface count:\t" << getFaceCount() << endl
		 << "\tread file time:\t" << htime.printElapseSec() << endl << endl;

	flog << "\t-----------------------------------------------" << endl 
	 	 << "\tread file successfully" << endl
		 << "\tfile name:\t" << filename << endl
		 << "\tvertex count:\t" << getVertexCount() << "\tface count:\t" << getFaceCount() << endl
		 << "\tread file time:\t" << htime.printElapseSec() << endl << endl;

	return true;
}

static SelfAdjointEigenSolver<Matrix3f> *solver;

static inline bool cmp(const int &a, const int &b)
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

	htime.setCheckPoint();

	/* - routines - */

	// init the first cluster
	for (i = 0; i < vertexCount; i ++) 
		if (vertices[i].clusterIndex != -1) 
			vc.addVertex(i, vertices[i]);

	cout << "\t-----------------------------------------------" << endl 
		 << "\tnon-referenced vertices count:\t" << vertexCount - vc.vIndices->size() << endl
		 << "\tminimum normal-vari factor:\t" << HSDVertexCluster2::MINIMUM_NORMAL_VARI << endl
		 << "\tvalid vertices count:\t" << vc.vIndices->size() << endl;

	flog << "\t-----------------------------------------------" << endl
		 << "\tnon-referenced vertices count:\t" << vertexCount - vc.vIndices->size() << endl
		 << "\tminimum normal-vari factor:\t" << HSDVertexCluster2::MINIMUM_NORMAL_VARI << endl
		 << "\tvalid vertices count:\t" << vc.vIndices->size() << endl;

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
			flog << "\tstop without reaching the target count because of unchanged cluster count" << endl;
			break;
		}
		
		if (clusters.empty()) {
			cerr << "#error: don't know why but the clusters heap have came to empty" << endl;
			flog << "#error: don't know why but the clusters heap have came to empty" << endl;
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
			flog << "#error: eigenvalues computing error" << endl;
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

#ifdef PRINT_DEBUG_INFO
		PrintHeap(fdebug, clusters);
#endif
	}

	//PrintHeap(cout, clusters);

	cout << "\tsimplification time:\t" << htime.printElapseSec() << endl << endl;

	flog << "\tsimplification time:\t" << htime.printElapseSec() << endl << endl;

	return true;
}

void HSpatialDivision2::clear()
{
	if (vertices) {
		delete[] vertices;
		vertices = NULL;
	}
	if (faces) {
		delete[] faces;
		faces = NULL;
	}
	if (vcArr2) {
		delete[] vcArr2;
		vcArr2 = NULL;
	}
	clusters.clear();
	degFaces.clear();
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
	cout << "\t-----------------------------------------------" << endl
		 << "\twrite simplified mesh successfully" << endl
		 << "\tfile name:\t" << filename << endl
		 << "\tvertex count:\t" << clusters.count() << "\tface count:\t" <<  degFaces.size() << endl
		 << "\twrite file time:\t\t" << htime.printElapseSec() << endl << endl;

	flog << "\t-----------------------------------------------" << endl
		 << "\twrite simplified mesh successfully" << endl
		 << "\tfile name:\t" << filename << endl
		 << "\tvertex count:\t\t" << clusters.count() << "\tface count:\t" <<  degFaces.size() << endl
		 << "\twrite file time:\t" << htime.printElapseSec() << endl << endl;

	return true;
}

void HSpatialDivision2::generateIndexedMesh()
{
	int i, vindex, i1, i2, i3;
	HSDVertexCluster2 sdc;
	HTripleIndex<Integer> tripleIndex;
	list<Integer>::iterator iter;

	for (i = 0; i < clusters.count(); i ++) {
		sdc = clusters.get(i);

		for (iter = sdc.vIndices->begin(); iter != sdc.vIndices->end(); iter ++) {
			vertices[*iter].clusterIndex = i;
		}
	}

	degFaces.clear();
	for (i = 0; i < faceCount; i ++) {
		i1 = vertices[faces[i].i].clusterIndex;
		i2 = vertices[faces[i].j].clusterIndex;
		i3 = vertices[faces[i].k].clusterIndex;

		if (i1 != i2 && i1 != i3 && i2 != i3) {
			tripleIndex.set(i1, i2, i3);
			degFaces.insert(tripleIndex);
		}
	}
}

void HSpatialDivision2::searchConnectivity(Integer vIndex, Integer clusterIndex) {

	list<Integer>::iterator iter;
	HTripleIndex<Integer> f;

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