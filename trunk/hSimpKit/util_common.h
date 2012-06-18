/*
 *  uitility & common classes
 *
 *  author: ht
 *  email : waytofall916@gmail.com
 */

#ifndef __UTIL_COMMON__
#define __UTIL_COMMON__

/* class defined */
class HVertex;
class HFaceFormula;
class HQEMatrix;

/* vertex */
class HVertex
{
public:
	HVertex() {}

	HVertex(float _x, float _y, float _z) {
		x = _x; y = _y; z = _z; }

	void set(float _x, float _y, float _z) {
		x = _x; y = _y; z = _z; }

	HVertex& operator+= (const HVertex v) {
		x += v.x; y += v.y; z += v.z; 
		return *this;
	}

	HVertex operator* (float n) const {
		HVertex v(x * n, y * n, z * n);		
		return v;
	}

	HVertex operator+ (HVertex &vertex) const {
		HVertex v(x + vertex.x, y + vertex.y, z + vertex.z);
		return v;
	}

	bool operator== (const HVertex &vertex) const {
		return x == vertex.x && y == vertex.y && z == vertex.z;
	}

	bool operator!= (const HVertex &vertex) const {
		return !operator==(vertex);
	}

public:
	float x, y, z;
};

/* calculate face formula */
class HFaceFormula
{
public:
	static void calcTriangleFaceFormula(HVertex _v1, HVertex _v2, HVertex _v3);
	static float calcTriangleFaceArea(HVertex _v1, HVertex _v2, HVertex _v3);

public:
	// parameters of a plane
	static float a, b, c, d;
};

/* quadric error metrics */
class HQEMatrix
{
public:
	HQEMatrix() {}

	void setZero();

	HQEMatrix(HSoupTriangle tri) {
		calcQem(tri); }

	HQEMatrix& operator+= (const HQEMatrix& m);

	void calcQem(HSoupTriangle tri)
	{
		HFaceFormula::calcTriangleFaceFormula(tri.v1, tri.v2, tri.v3);
		calcQem(HFaceFormula::a, HFaceFormula::b, HFaceFormula::c, HFaceFormula::d);
	}

	HQEMatrix operator* (float f) const;
	HQEMatrix& operator*= (float f);

	void calcQem(float a, float b, float c, float d);
	bool calcRepresentativeVertex(HVertex& vertex);

public:
	float a11, a12, a13, a14;
	float      a22, a23, a24;
	float           a33, a34;
	float                a44;
};

#endif //__UTIL_COMMON__
