/*
	triangle soup stream class
	read/write triangle soup
	stream file
	author: ht
*/

#include "tri_soup_stream.h"

int TriSoupStream::openForWrite(const char *filename)
{
	fout.open(filename, ios::out | ios::binary);

	fout.write((char*)&max_x, sizeof(float));
	fout.write((char*)&min_x, sizeof(float));
	fout.write((char*)&max_y, sizeof(float));
	fout.write((char*)&min_y, sizeof(float));
	fout.write((char*)&max_z, sizeof(float));
	fout.write((char*)&min_z, sizeof(float));

	if (fout.good())
		return 1;
	else
		return 0;
}

int TriSoupStream::writeFloat(float x1, float y1, float z1, float x2, float y2, float z2, float x3, float y3, float z3)
{
	//fout << x1 << y1 << z1 << x2 << y2 << z2 << x3 << y3 << z3;

	fout.write((char*)&x1, sizeof(float));
	fout.write((char*)&y1, sizeof(float));
	fout.write((char*)&z1, sizeof(float));
	fout.write((char*)&x2, sizeof(float));
	fout.write((char*)&y2, sizeof(float));
	fout.write((char*)&z2, sizeof(float));
	fout.write((char*)&x3, sizeof(float));
	fout.write((char*)&y3, sizeof(float));
	fout.write((char*)&z3, sizeof(float));

	if (fout.good())
		return 1;
	else
		return 0;
}

int TriSoupStream::closeForWrite()
{
	fout.close();
	if (fout.good())
		return 1;
	else
		return 0;
}

int TriSoupStream::openForRead(const char *filename)
{
	fin.open(filename, ios::binary | ios::in);

	fin.read((char*)&max_x, sizeof(float));
	fin.read((char*)&min_x, sizeof(float));
	fin.read((char*)&max_y, sizeof(float));
	fin.read((char*)&min_y, sizeof(float));
	fin.read((char*)&max_z, sizeof(float));
	fin.read((char*)&min_z, sizeof(float));
	
	if (fin.good())
		return 1;
	else
		return 0;
}

int TriSoupStream::readNext()
{
	fin.read((char*)vertices, sizeof(float) * 9);

	if (fin.eof())
		return 0;
	else
		return 1;
}

int TriSoupStream::closeForRead()
{
	fin.close();

	if (fin.good())
		return 1;
	else
		return 0;
}

float TriSoupStream::getFloat(unsigned int i, unsigned int j)
{
	if (i < 0 || i > 2 || j < 0 || j > 2)
		return 0;
	
	return vertices[i * 3  + j];
}

void TriSoupStream::setBoundBox(float _max_x, float _min_x, float _max_y, float _min_y, float _max_z, float _min_z)
{
	max_x = _max_x;
	min_x = _min_x;
	max_y = _max_y;
	min_y = _min_y;
	max_z = _max_z;
	min_z = _min_z;
}

float TriSoupStream::getMaxX()
{
	return max_x;
}

float TriSoupStream::getMinX()
{
	return min_x;
}

float TriSoupStream::getMaxY()
{
	return max_y;
}

float TriSoupStream::getMinY()
{
	return min_y;
}

float TriSoupStream::getMaxZ()
{
	return max_z;
}

float TriSoupStream::getMinZ()
{
	return min_z;
}