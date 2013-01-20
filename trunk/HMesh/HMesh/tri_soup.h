/*
	triangle soup container
	author: ht
*/

#ifndef __TRI_SOUP_CONTAINER__
#define __TRI_SOUP_CONTAINER__

#include "tri_soup_stream.h"

typedef struct
{
	float x, y, z;
} TripleFloat;

typedef struct
{
	TripleFloat vert1;
	TripleFloat vert2;
	TripleFloat vert3;
} SoupTriangle;

class TriangleSoupContainer
{
private:
	// variables concerning triangle soups
	SoupTriangle *p_tri_soup;
	unsigned int tri_soup_size;
	unsigned int tri_soup_count;
	const unsigned int init_size;
	TriSoupStream sstream;

public:
	TriangleSoupContainer();
	~TriangleSoupContainer();
	void init();
	void clear();
	bool read(const char* filename);
	SoupTriangle operator()(unsigned int i);
	unsigned int count();
	TriSoupStream* getTriSStream();
};

#endif //__TRI_SOUP_CONTAINER__