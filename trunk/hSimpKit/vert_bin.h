/*
	manipulations for the vertex binary file
	there is a least recent used hash cache
	for reading vertices

	for writing, this is simple stream writing

	the hash bucket with lru list is like this

	_____________        _____          _____
	|bucket_head| ---->  |   | -> .. -> |   | -> Null
	_____________        _____          _____
	|bucket_head| ---->  |   | -> .. -> |   | -> Null
	_____________        
	|bucket_head| ---->  Null
	
	...
	_____________        _____          _____
	|bucket_head| ---->  |   | -> .. -> |   | -> Null

	while all the nodes in the buckets are linked
	based on the time they are accessed, thus the
	newly accessed node are inserted into the head
	of the lru list, if it has already existed in
	the list, it will be deleted from the list first

	author: houtao
*/

#ifndef __VERT_BIN__
#define __VERT_BIN__

#include <fstream>
#include <iostream>
using namespace std;

class VertexBinary
{
	typedef struct __Vertex
	{
		float x, y, z;
	} __Vertex;

	typedef struct __CacheUnit
	{
		struct __CacheUnit *bucket_prev; // bucket pointer
		struct __CacheUnit *bucket_next;
		struct __CacheUnit *use_prev; // lru pointer
 		struct __CacheUnit *use_next;
		__Vertex vert;
		unsigned int index;
	} __CacheUnit;

public:
	VertexBinary();
	~VertexBinary();

	/* write */
	int openForWrite(const char* filename);
	int closeWriteFile();
	int writeVertexFloat(float x, float y, float z);

	/* read */
	int initCache(unsigned int size);
	int openForRead(const char* filename);
	int indexedRead(unsigned int index);
	int closeReadFile();
	float getXFloat();
	float getYFloat();
	float getZFloat();

	void writeCacheDebug();

private:
	void insertBucketList(__CacheUnit **head, __CacheUnit *new_unit);
	void deleteBucketList(__CacheUnit **head, __CacheUnit *unit);
	void insertLruList(__CacheUnit *new_unit);
	void deleteLruList(__CacheUnit *unit);
	int indexedReadFromFile(unsigned int index, float &x, float &y, float &z);

private:
	ifstream fin;
	ofstream fout;
	unsigned int cache_size;
	unsigned int cache_count;
	__CacheUnit **cache_bucket;
	__CacheUnit *lru_head;
	__CacheUnit *lru_tail;
	__CacheUnit *current_unit;

public:
	unsigned int read_count;
	unsigned int hit_count;
};

#endif