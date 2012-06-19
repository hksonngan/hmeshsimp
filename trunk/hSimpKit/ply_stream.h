/*
 *  ply read & write
 * 
 *  author : ht
 *  email  : waytofall916@gmail.com
 */

#ifndef __PLY_STREAM__
#define __PLY_STREAM__

#include "ply/ply.h"
#include "util_common.h"

typedef struct Vertex {
	float x,y,z;
	void *other_props;       /* other properties */
} Vertex;

typedef struct Face {
	unsigned char nverts;    /* number of vertex indices in list */
	int *verts;              /* vertex index list */
	void *other_props;       /* other properties */
} Face;

class PlyStream
{
public:
	PlyStream();
	bool openForRead(char *filename);
	bool close();
	bool nextVertex(HVertex &v);
	bool nextFace(HTripleIndex &f);

	Integer getVertexCount() { return vCount; };
	Integer getFaceCount() { return fCount; };

private:
	PlyFile *ply;
	Integer vCount;
	Integer fCount;
	Integer readVCount;
	Integer readFCount;
	Vertex vertex;
	Face face;
};

#endif //__PLY_STREAM__