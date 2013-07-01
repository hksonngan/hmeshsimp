/*
   Marching Tetrahedron Implementation
   Modified from http://paulbourke.net/geometry/polygonise/
 */

#ifndef _DEVICE_MARCH_TETRA_H_
#define _DEVICE_MARCH_TETRA_H_

#define MINI 0.00001

template <typename T>
__forceinline__ __device__
T ABS(const T& v) {
	return v >= 0? v: -v;
}

__device__
void VertexInterp(
	const float &isolevel, const XYZ &p1, const XYZ &p2, 
	const float &valp1, const float &valp2, float *p
){
	float mu;

	if (ABS(isolevel-valp1) < MINI) {
		p[0] = p1.x;
		p[1] = p1.y;
		p[2] = p1.z;
	}
	if (ABS(isolevel-valp2) < MINI) {
		p[0] = p2.x;
		p[1] = p2.y;
		p[2] = p2.z;
	}
	if (ABS(valp1-valp2) < MINI) {
		p[0] = p1.x;
		p[1] = p1.y;
		p[2] = p1.z;
	}

	mu = (isolevel - valp1) / (valp2 - valp1);
	p[0] = p1.x + mu * (p2.x - p1.x);
	p[1] = p1.y + mu * (p2.y - p1.y);
	p[2] = p1.z + mu * (p2.z - p1.z);
}

template <typename T>
__forceinline__ __device__
void copy3(T *dst, const T* src) {
	dst[0] = src[0];
	dst[1] = src[1];
	dst[2] = src[2];
}


/*
   Polygonise a tetrahedron given its vertices within a cube
   This is an alternative algorithm to polygonise grid.
   It results in a smoother surface but more triangular facets.

                      + 0
                     /|\
                    / | \
                   /  |  \
                  /   |   \
                 /    |    \
                /     |     \
               +------|------+ 1
              3 \     |     /
                 \    |    /
                  \   |   /
                   \  |  /
                    \ | /
                     \|/
                      + 2

   It's main purpose is still to polygonise a gridded dataset and
   would normally be called 6 times, one for each tetrahedron making
   up the grid cell.
   Given the grid labelling as in PolygniseGrid one would call
      PolygoniseTri(grid,iso,triangles,0,2,3,7);
      PolygoniseTri(grid,iso,triangles,0,2,6,7);
      PolygoniseTri(grid,iso,triangles,0,4,6,7);
      PolygoniseTri(grid,iso,triangles,0,6,1,2);
      PolygoniseTri(grid,iso,triangles,0,6,1,4);
      PolygoniseTri(grid,iso,triangles,5,6,1,4);
*/
// I reorder the vertex sequence so that the light could be properly rendered
// in the figure above, the edge 02 is in front of edge 31, which means that
// 31 is blocked by 02
//  -- Ht

__device__
unsigned char PolygoniseTri(Tetra &g, const float &iso, float *tri)
{
	unsigned char ntri = 0;
	unsigned char triindex;

	/*
		Determine which of the 16 cases we have given which vertices
		are above or below the isosurface
	*/
	triindex = 0;
	if (g.val[0] < iso) triindex |= 1;
	if (g.val[1] < iso) triindex |= 2;
	if (g.val[2] < iso) triindex |= 4;
	if (g.val[3] < iso) triindex |= 8;

	/* Form the vertices of the triangles for each case */
	switch (triindex) {
	case 0x00:
	case 0x0F:
		break;
	case 0x0E:
		VertexInterp(iso,g.p[0],g.p[1],g.val[0],g.val[1],tri);
		VertexInterp(iso,g.p[0],g.p[2],g.val[0],g.val[2],tri+3);
		VertexInterp(iso,g.p[0],g.p[3],g.val[0],g.val[3],tri+6);
		ntri++;
		break;
	case 0x01:
		VertexInterp(iso,g.p[0],g.p[1],g.val[0],g.val[1],tri);
		VertexInterp(iso,g.p[0],g.p[3],g.val[0],g.val[3],tri+3);
		VertexInterp(iso,g.p[0],g.p[2],g.val[0],g.val[2],tri+6);
		ntri++;
		break;
	case 0x0D:
		VertexInterp(iso,g.p[1],g.p[0],g.val[1],g.val[0],tri);
		VertexInterp(iso,g.p[1],g.p[3],g.val[1],g.val[3],tri+3);
		VertexInterp(iso,g.p[1],g.p[2],g.val[1],g.val[2],tri+6);
		ntri++;
		break;
	case 0x02:
		VertexInterp(iso,g.p[1],g.p[0],g.val[1],g.val[0],tri);
		VertexInterp(iso,g.p[1],g.p[2],g.val[1],g.val[2],tri+3);
		VertexInterp(iso,g.p[1],g.p[3],g.val[1],g.val[3],tri+6);
		ntri++;
		break;
	case 0x0C:
		VertexInterp(iso,g.p[0],g.p[2],g.val[0],g.val[2],tri);
		VertexInterp(iso,g.p[0],g.p[3],g.val[0],g.val[3],tri+3);
		VertexInterp(iso,g.p[1],g.p[3],g.val[1],g.val[3],tri+6);
		ntri++;
		copy3(tri+9, tri+6);
		VertexInterp(iso,g.p[1],g.p[2],g.val[1],g.val[2],tri+12);
		copy3(tri+15, tri);
		ntri++;
		break;
	case 0x03:
		VertexInterp(iso,g.p[0],g.p[3],g.val[0],g.val[3],tri);
		VertexInterp(iso,g.p[0],g.p[2],g.val[0],g.val[2],tri+3);
		VertexInterp(iso,g.p[1],g.p[3],g.val[1],g.val[3],tri+6);
		ntri++;
		VertexInterp(iso,g.p[1],g.p[2],g.val[1],g.val[2],tri+9);
		copy3(tri+12, tri+6);
		copy3(tri+15, tri+3);
		ntri++;
		break;
	case 0x0B:
		VertexInterp(iso,g.p[2],g.p[0],g.val[2],g.val[0],tri);
		VertexInterp(iso,g.p[2],g.p[1],g.val[2],g.val[1],tri+3);
		VertexInterp(iso,g.p[2],g.p[3],g.val[2],g.val[3],tri+6);
		ntri++;
		break;
	case 0x04:
		VertexInterp(iso,g.p[2],g.p[0],g.val[2],g.val[0],tri);
		VertexInterp(iso,g.p[2],g.p[3],g.val[2],g.val[3],tri+3);
		VertexInterp(iso,g.p[2],g.p[1],g.val[2],g.val[1],tri+6);
		ntri++;
		break;
	case 0x0A:
		VertexInterp(iso,g.p[0],g.p[1],g.val[0],g.val[1],tri);
		VertexInterp(iso,g.p[2],g.p[3],g.val[2],g.val[3],tri+3);
		VertexInterp(iso,g.p[0],g.p[3],g.val[0],g.val[3],tri+6);
		ntri++;
		copy3(tri+9, tri);
		VertexInterp(iso,g.p[1],g.p[2],g.val[1],g.val[2],tri+12);
		copy3(tri+15, tri+3);
		ntri++;
		break;
	case 0x05:
		VertexInterp(iso,g.p[0],g.p[1],g.val[0],g.val[1],tri);
		VertexInterp(iso,g.p[0],g.p[3],g.val[0],g.val[3],tri+3);
		VertexInterp(iso,g.p[2],g.p[3],g.val[2],g.val[3],tri+6);
		ntri++;
		copy3(tri+9, tri);
		copy3(tri+12, tri+6);
		VertexInterp(iso,g.p[1],g.p[2],g.val[1],g.val[2],tri+15);
		ntri++;
		break;
	case 0x09:
		VertexInterp(iso,g.p[0],g.p[1],g.val[0],g.val[1],tri);
		VertexInterp(iso,g.p[1],g.p[3],g.val[1],g.val[3],tri+3);
		VertexInterp(iso,g.p[2],g.p[3],g.val[2],g.val[3],tri+6);
		ntri++;
		copy3(tri+9, tri);
		copy3(tri+12, tri+6);
		VertexInterp(iso,g.p[0],g.p[2],g.val[0],g.val[2],tri+15);
		ntri++;
		break;
	case 0x06:
		VertexInterp(iso,g.p[0],g.p[1],g.val[0],g.val[1],tri);
		VertexInterp(iso,g.p[2],g.p[3],g.val[2],g.val[3],tri+3);
		VertexInterp(iso,g.p[1],g.p[3],g.val[1],g.val[3],tri+6);
		ntri++;
		copy3(tri+9, tri);
		VertexInterp(iso,g.p[0],g.p[2],g.val[0],g.val[2],tri+12);
		copy3(tri+15, tri+3);
		ntri++;
		break;
	case 0x07:
		VertexInterp(iso,g.p[3],g.p[0],g.val[3],g.val[0],tri);
		VertexInterp(iso,g.p[3],g.p[2],g.val[3],g.val[2],tri+3);
		VertexInterp(iso,g.p[3],g.p[1],g.val[3],g.val[1],tri+6);
		ntri++;
		break;
	case 0x08:
		VertexInterp(iso,g.p[3],g.p[0],g.val[3],g.val[0],tri);
		VertexInterp(iso,g.p[3],g.p[1],g.val[3],g.val[1],tri+3);
		VertexInterp(iso,g.p[3],g.p[2],g.val[3],g.val[2],tri+6);
		ntri++;
		break;
	}

	return(ntri);
}

__device__
const unsigned char& PolygoniseTriGetCount(const Tetra &g, const float &iso)
{
	unsigned char triindex;
	unsigned char ntri = 0;

	/*
		Determine which of the 16 cases we have given which vertices
		are above or below the isosurface
	*/
	triindex = 0;
	if (g.val[0] < iso) triindex |= 1;
	if (g.val[1] < iso) triindex |= 2;
	if (g.val[2] < iso) triindex |= 4;
	if (g.val[3] < iso) triindex |= 8;

	/* Form the vertices of the triangles for each case */
	switch (triindex) {
	case 0x00:
	case 0x0F:
		break;
	case 0x0E:
	case 0x01:
	case 0x0D:
	case 0x02:
	case 0x07:
	case 0x08:
	case 0x0B:
	case 0x04:
		ntri=1;
		break;
	case 0x0C:
	case 0x03:
	case 0x0A:
	case 0x05:
	case 0x09:
	case 0x06:
		ntri=2;
		break;
	}

	return(ntri);
}

#endif