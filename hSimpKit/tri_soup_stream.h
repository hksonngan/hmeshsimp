/*
	triangle soup stream class
	read/write triangle soup
	stream file
	author: ht
*/

#ifndef __TRI_SOUP_STREAM__
#define __TRI_SOUP_STREAM__

#include <iostream>
#include <fstream>

using namespace std;

class TriSoupStream
{
public:
	/* write */
	int openForWrite(const char *filename);
	int writeFloat(float x1, float y1, float z1, float x2, float y2, float z2, float x3, float y3, float z3);
	void setBoundBox(float _max_x, float _min_x, float _max_y, float _min_y, float _max_z, float _min_z);
	int closeForWrite();

	/* read */
	int openForRead(const char *filename);
	int readNext();
	/* i: triangle index j: x y z index */
	float getFloat(unsigned int i, unsigned int j);
	float getMaxX();
	float getMinX();
	float getMaxY();
	float getMinY();
	float getMaxZ();
	float getMinZ();
	int closeForRead();

private:
	ofstream fout;
	ifstream fin;
	float vertices[9];
	/* bounding box */
	float max_x, min_x;
	float max_y, min_y;
	float max_z, min_z;
};

#endif //__TRI_SOUP_STREAM__