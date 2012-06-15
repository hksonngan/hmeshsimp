/*
	manipulations for the vertex binary file
	author: houtao
*/

#include "vert_bin.h"
#include <fstream>
#include <iostream>

// if dump the cache to the file
//#define _WRITE_CACHE_DEBUG

VertexBinary::VertexBinary()
{
	cache_bucket = NULL;
	cache_size = 0;
	cache_count = 0;
}

VertexBinary::~VertexBinary()
{
	if (cache_bucket)
	{
		delete[] cache_bucket;
	}
}

int VertexBinary::openForWrite(const char *filename)
{
	fout.open(filename, ios::out | ios::binary);

	if(fout.good())
		return 1;
	else
		return 0;
}

int VertexBinary::writeVertexFloat(float x, float y, float z)
{
	if (!fout.good())
	{
		return 0;
	}

	fout.write((char*)&x, sizeof(float));
	fout.write((char*)&y, sizeof(float));
	fout.write((char*)&z, sizeof(float));

	return 1;
}

int VertexBinary::openForRead(const char* filename)
{
	fin.open(filename, ios::binary | ios::in);
	if (fin.good())
		return 1;
	else
		return 0;
}

int VertexBinary::indexedRead(unsigned int index)
{
	if (cache_size <= 0)
	{
		cerr << "#error : no cache while indexed reading" << endl;
		return 0;
	}
	
	read_count ++;

	int hash_index = index % cache_size;
	__CacheUnit *hit_unit = NULL;

	// check if the cache hit
	if(cache_bucket[hash_index])
	{
		__CacheUnit *bucket_head;
		for(bucket_head = cache_bucket[hash_index]; bucket_head; bucket_head = bucket_head->bucket_next)
		{
			if (bucket_head->index == index)
			{
				hit_unit = bucket_head;
				break;
			}
		}
	}

	// the the cache has the queried unit
	if (hit_unit)
	{
		deleteLruList(hit_unit);
		insertLruList(hit_unit);
		current_unit = hit_unit;
		hit_count ++;
		return 1;
	}

	// the cache doesn't store the queried unit

	// check if needed to delete the least recent used unit
	if (cache_count == cache_size)
	{
		// delete the tail unit from the two lists
		hit_unit = lru_tail;
		deleteBucketList(&cache_bucket[hit_unit->index % cache_size], hit_unit);
		deleteLruList(hit_unit);
	}
	else
	{
		hit_unit =  new __CacheUnit();
		cache_count ++;
	}

	hit_unit->index = index;
	indexedReadFromFile(index, hit_unit->vert.x, hit_unit->vert.y, hit_unit->vert.z);
	insertBucketList(&cache_bucket[hash_index], hit_unit);
	insertLruList(hit_unit);
	current_unit = hit_unit;

	#ifdef _WRITE_CACHE_DEBUG
	writeCacheDebug();
	#endif

	return 1;
}

int VertexBinary::initCache(unsigned int size)
{
	if(size <= 0)
		return 0;

	cache_size = size;
	cache_bucket = new __CacheUnit*[cache_size];

	for(unsigned int i = 0; i < cache_size; i ++)
	{
		cache_bucket[i] = NULL;
	}

	cache_count = 0;
	lru_head = NULL;
	lru_tail = NULL;

	read_count = 0; 
	hit_count = 0;

	return 1;
}

// insert from head
void VertexBinary::insertBucketList(__CacheUnit **head, __CacheUnit *new_unit)
{
	new_unit->bucket_prev = NULL;
	new_unit->bucket_next = *head;
	if (*head)
	{
		(*head)->bucket_prev = new_unit;
	}
	(*head) = new_unit;
}

// delete from any place
void VertexBinary::deleteBucketList(__CacheUnit **head, __CacheUnit *unit)
{
	if(*head == NULL)
		return;

	if(*head == unit)
	{
		*head = unit->bucket_next;
	}
	if (unit->bucket_prev)
	{
		unit->bucket_prev->bucket_next = unit->bucket_next;
	}
	if (unit->bucket_next)
	{
		unit->bucket_next->bucket_prev = unit->bucket_prev;
	}
}

void VertexBinary::insertLruList(__CacheUnit *new_unit)
{
	new_unit->use_prev = NULL;
	new_unit->use_next = lru_head;
	if (lru_head)
	{
		lru_head->use_prev = new_unit;
	}

	if (lru_head == NULL)
	{
		lru_tail = new_unit;
	}

	lru_head = new_unit;
}

// delete the lru list from the tail
void VertexBinary::deleteLruList(__CacheUnit *unit)
{
	if (lru_head == NULL)
		return;

	if (unit->use_next)
	{
		unit->use_next->use_prev = unit->use_prev;
	}
	if (unit->use_prev)
	{
		unit->use_prev->use_next = unit->use_next;
	}
	
	if (lru_head == unit)
	{
		lru_head = unit->use_next;
	}
	if (lru_tail == unit)
	{
		lru_tail = unit->use_prev;
	}
}

int VertexBinary::indexedReadFromFile(unsigned int index, float &x, float &y, float &z)
{
	if (!fin.good())
	{
		return 0;
	}

	fin.seekg(index * 3 * sizeof(float));
	fin.read((char*)&x, sizeof(float));
	fin.read((char*)&y, sizeof(float));
	fin.read((char*)&z, sizeof(float));

	return 1;
}

int VertexBinary::closeWriteFile()
{
	fout.close();

	return 1;
}

int VertexBinary::closeReadFile()
{
	fin.close();

	return 1;
}

float VertexBinary::getXFloat()
{
	return current_unit->vert.x;
}

float VertexBinary::getYFloat()
{
	return current_unit->vert.y;
}

float VertexBinary::getZFloat()
{
	return current_unit->vert.z;
}

void VertexBinary::writeCacheDebug()
{
	int i, c1, c2;
	using namespace std;
	ofstream fout ("cache_dump.txt", ios::out | ios::app);
	__CacheUnit *pUnit;

	for(i = 0, c1 = 0; i < cache_size; i ++)
	{
		if (cache_bucket[i] == NULL)
			continue;

		fout << "bucket #" << i << ": ";
		for(pUnit = cache_bucket[i], c2 = 0; pUnit; pUnit = pUnit->bucket_next, c2 ++)
		{
			fout << pUnit->index << " ";
		}
		fout << "count: " << c2 << " " << endl;
		c1 += c2;
	}

	fout << "total count: " << c1 << endl << "lru list: ";
	for(c1 = 0, pUnit = lru_head; pUnit; pUnit = pUnit->use_next, c1 ++)
	{
		fout << pUnit->index << " ";
	}

	fout << endl << "lru count: " << c1 << endl << endl;

	fout.close();
}