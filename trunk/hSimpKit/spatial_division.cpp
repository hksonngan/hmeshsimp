#include "spatial_division.h"
#include "util_common.h"
#include "math/vec3.h"

/* -- spatial division vertex cluster -- */

HSDVertexCluster::HSDVertexCluster()
{
	awQ.setZero();
	awN.set(0.0, 0.0, 0.0);
	vIndices = NULL;
	area = 0;
}

void HSDVertexCluster::addVertex(Integer i, HQEMatrix _awQ, HNormal _awN, float _area)
{
	//if (vCount + 1 >= vSize)
	//{
	//	Integer *new_v = new Integer[vSize * 2];
	//	memcpy(new_v, vIndices, sizeof(Integer) * vSize);
	//	delete[] vIndices;
	//	vIndices = new_v;
	//	vSize *= 2;
	//}

	if (vIndices == NULL)
	{
		vIndices = new vector<Integer>();
	}

	vIndices->push_back(i);

	this->awN += _awN;
	this->awQ += _awQ;
	this->area += _area;
}

bool HSDVertexCluster::operator< (const HSDVertexCluster &vc) const
{
	Vec3<float> n1(this->awN.x, this->awN.y, this->awN.z),
		n2(vc.awN.x, vc.awN.y, vc.awN.z);

	n1 /= area;
	n2 /= area;

	float l1 = 1 - n1.Length();
	float l2 = 1 - n2.Length();

	return l1 < l2;
}

/* -- spatial division class -- */

HSpatialDivision::HSpatialDivision()
:clusters(1, INIT_HEAP_VOL)
{

}

inline void HSpatialDivision::addVertex(HVertex v)
{
	HSDVertex sdv;
	sdv.set(v.x, v.y, v.z);

	vertices.push_back(sdv);
}

void HSpatialDivision::addFace(HTripleIndex i3)
{
	faces.push_back(i3);

	//float area = HFaceFormula::calcTriangleFaceArea(vertices[i3.i], vertices[i3.j], vertices[i3.k]);

	Vec3<float> v1, v2, v3;
	v1.Set(vertices[i3.i].x, vertices[i3.i].y, vertices[i3.i].z);
	v1.Set(vertices[i3.j].x, vertices[i3.j].y, vertices[i3.j].z);
	v1.Set(vertices[i3.k].x, vertices[i3.k].y, vertices[i3.k].z);
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

void HSpatialDivision::divide(int target_count)
{
	HSDVertexCluster vc1, vc2, vc3, vc4, vc5, vc6, vc7, vc8;
	int i;

	//init the first cluster
	for (i = 0; i < this->vertices.size(); i ++)
	{
		vc1.addVertex(i, vertices[i].awQ, vertices[i].awN, vertices[i].area);
	}
	clusters.addElement(vc1);

	// subdivide until the divided clusters reach the target count
	while(clusters.count() < target_count)
	{
		
	}
}

void HSpatialDivision::toPly()
{

}