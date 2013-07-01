/*
 *  Triangle Soup Container
 *
 *  Author: Ht
 *  Email : waytofall916 at gmail dot com
 *
 *  Copyright (C) Ht. All rights reserved.
 */

#ifndef __TRI_SOUP_CONTAINER__
#define __TRI_SOUP_CONTAINER__

#include "tri_soup_stream.h"

// A vertex
typedef struct
{
	float x, y, z;
} TripleFloat;

// Struct that holds the a triangle
typedef struct
{
	TripleFloat vert1;
	TripleFloat vert2;
	TripleFloat vert3;
} SoupTriangle;

// Triangle Soup Container
// A dynamic array containing 'SoupTriangle' type elements
// A better way is to simply wrap 'vector' or even 'HDynamArray'
// However, it is an early version
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