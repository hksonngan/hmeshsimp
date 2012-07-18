/*
 *	Manipulations for the LRU cache file.
 *  !! The file seek policy may not be platform independent now !!
 *
 *	There is a least recent used hash cache for reading, the hashed value is the index 
 *	of the value. So the using scenario is that there are some values that is too big to
 *	fit in the main memory while residing one by one in the binary file, and there are
 *	some reading mechanics that read the value based on the location that they are residing 
 *	in the disk (the index)
 *
 *	For writing, this is simple binary(!) stream writing.
 *
 *	The hash bucket with LRU list is like this
 *	_____________        _____          _____
 *	|bucket_head| ---->  |   | -> .. -> |   | -> Null
 *	_____________        _____          _____
 *	|bucket_head| ---->  |   | -> .. -> |   | -> Null
 *	_____________        
 *	|bucket_head| ---->  Null
 *	
 *	...
 *	_____________        _____          _____
 *	|bucket_head| ---->  |   | -> .. -> |   | -> Null
 *
 *	While all the nodes in the buckets are linked based on the time they are accessed, thus the
 *	newly accessed node are inserted into the head of the LRU list, if it has already existed in
 *	the list, it will be deleted from the list first.
 *
 *  Author: Ht
 *  Email : waytofall916 at gmail dot com
 * 
 *  Copyright (C) Ht-waytofall. All rights reserved.
 */


#ifndef __LRU_CACHE__
#define __LRU_CACHE__

#include <fstream>
#include <iostream>
#include <stdio.h>

using std::ifstream;
using std::ofstream;
using std::streampos;
using std::cerr;
using std::endl;

#define LRU_SUCCESS	1
#define LRU_FAIL	0


/* ============================= & DEFINITION & ============================== */

/* if dump the cache to the file */
//#define _WRITE_CACHE_DEBUG

/* you should inherit this class if you want to
   use the LRU cache, and make this class as the
   template argument for the class template 'LRUCache' */
class LRUVal {
public:
	// binary(!) read
	bool read(ifstream& fin) { return true; }
	bool read(FILE *fp) { return true; }
	// binary(!) write
	bool write(ofstream& fout) { return true; }
	// hash the index
	static unsigned int hash(unsigned int index) { return index; }
	// the size of the data stored in file
	static size_t size() { return 0; }
};

/* the cache unit type */
template <class ValType>
class __CacheUnit
{
public:
	__CacheUnit<ValType> *bucket_prev;	// bucket pointer
	__CacheUnit<ValType> *bucket_next;
	__CacheUnit<ValType> *use_prev;		// lru pointer
	__CacheUnit<ValType> *use_next;
	ValType val;
	unsigned int index;					// global index of the cached unit
};

/* least recent used cache */
template <class ValType> /* type of the value contained in the cache, this type must be a derivative of LRUVal */
class LRUCache
{
	/// abandoned
	//typedef struct __Vertex
	//{
	//	float x, y, z;
	//} __Vertex;
	///

public:
	LRUCache();
	~LRUCache();

	/* all the int returned value denotes success */ 

	/* ~ write ~ */
	int openForWrite(const char* filename);
	int closeWriteFile();
	inline int writeVal(ValType &val);

	/* ~ read ~ */
	int openForRead(const char* filename);
	void setReadFile(ifstream *_fin, streampos _start_pos);
	void setReadFile(FILE *_fp, fpos_t _start_pos);
	int initCache(unsigned int _buckets, unsigned int _cache_size);
	/* read the value of the given index */
	int indexedRead(
		unsigned int index,	/*given index*/ 
		ValType &val);		/*returned value*/
	int closeReadFile();

	/* for debug */
	void writeCacheDebug();
	
	/// abandoned
	//int writeVertexFloat(float x, float y, float z);
	//int indexedRead(unsigned int index);
	//float getXFloat();
	//float getYFloat();
	//float getZFloat();
	///

private:
	void insertBucketList(__CacheUnit<ValType> **head, __CacheUnit<ValType> *new_unit);
	void deleteBucketList(__CacheUnit<ValType> **head, __CacheUnit<ValType> *unit);
	void insertLruList(__CacheUnit<ValType> *new_unit);
	void deleteLruList(__CacheUnit<ValType> *unit);

	inline int indexedReadFromFile(unsigned int index, ValType &val);
	
	/// abandoned
	//int indexedReadFromFile(unsigned int index, float &x, float &y, float &z);
	///

private:
	unsigned int			cache_size;
	unsigned int			cache_count;
	unsigned int			bucket_count;
	__CacheUnit<ValType>	**cache_bucket;
	__CacheUnit<ValType>	*lru_head;
	__CacheUnit<ValType>	*lru_tail;

	bool		cfile;
	ifstream	*fin;
	ifstream	&fin_obj;
	streampos	start_pos;
	FILE		*fp;
	fpos_t		start_pos_c;

	ofstream	fout;
	
	/// abandoned
	//__CacheUnit *current_unit;
	///

public:
	unsigned int	read_count;
	unsigned int	hit_count;
};


/* ============================= & IMPLEMENTATION & ============================== */

template <class ValType>
LRUCache<ValType>::LRUCache()
{
	cache_bucket = NULL;
	lru_head = NULL;
	lru_tail = NULL;
	cache_size = 0;
	bucket_count = 0;
	cache_count = 0;
}

template <class ValType>
LRUCache<ValType>::~LRUCache()
{
	if (cache_bucket)
	{
		delete[] cache_bucket;
	}
}

template <class ValType>
int LRUCache<ValType>::openForWrite(const char *filename)
{
	fout.open(filename, ios::out | ios::binary);

	if(fout.good())
		return LRU_SUCCESS;
	else
		return LRU_FAIL;
}

/// abandoned
//template <class ValType, class Equal>
//int LRUCache<ValType, Equal>::writeVertexFloat(float x, float y, float z)
//{
//	if (!fout.good())
//	{
//		return 0;
//	}
//
//	fout.write((char*)&x, sizeof(float));
//	fout.write((char*)&y, sizeof(float));
//	fout.write((char*)&z, sizeof(float));
//
//	return 1;
//}
///

template <class ValType>
int LRUCache<ValType>::writeVal(ValType &val) {

	if (val.write(fout)) 
		return LRU_SUCCESS;
	return LRU_FAIL;
}

template <class ValType, class Equal>
int LRUCache<ValType, Equal>::openForRead(const char* filename)
{
	fin_obj.open(filename, ios::binary | ios::in);
	fin = &fin_obj;
	start_pos = 0 /*fstream::beg*/;
	if (fin_obj.good())
		return LRU_SUCCESS;
	else
		return LRU_FAIL;
}

template <class ValType>
void LRUCache<ValType>::setReadFile(ifstream *_fin, streampos _start_pos) {

	cfile = false;
	fin = _fin;
	start_pos = _start_pos;
}

template <class ValType>
void LRUCache<ValType>::setReadFile(FILE *_fp, fpos_t _start_pos) {

	cfile = true;
	fp = _fp;
	start_pos_c = _start_pos;
}

template <class ValType>
int LRUCache<ValType>::initCache(unsigned int _buckets, unsigned int _cache_size)
{
	if (_buckets == 0 || _cache_size == 0) {
		cerr << "\t#ERROR in LRUCache::initCache: buckets count or cache size should not be zero" << endl;
		return LRU_FAIL;
	}

	bucket_count = _buckets;
	cache_size = _cache_size;
	cache_bucket = new __CacheUnit<ValType>*[bucket_count];

	for(int i = 0; i < bucket_count; i ++)
		cache_bucket[i] = NULL;

	cache_count = 0;
	lru_head = NULL;
	lru_tail = NULL;

	read_count = 0; 
	hit_count = 0;

	return LRU_SUCCESS;
}

template <class ValType>
int LRUCache<ValType>::indexedRead(unsigned int index, ValType &val)
{
	if (cache_size == 0 || bucket_count == 0)
	{
		cerr << "\t#ERROR in LRUCache::indexedRead : no cache while indexed reading" << endl;
		return LRU_FAIL;
	}

	read_count ++;

	/* the bucket index of the value of the given index */
	int bucket_index = ValType::hash(index) % bucket_count;
	__CacheUnit<ValType> *hit_unit = NULL;

	/* check if the cache hit */
	if(cache_bucket[bucket_index])
	{
		__CacheUnit<ValType> *bucket_node;
		for(bucket_node = cache_bucket[bucket_index]; bucket_node; bucket_node = bucket_node->bucket_next) 
		{
			if (bucket_node->index == index) {
				hit_unit = bucket_node;
				break;
			}
		}
	}

	/* the the cache has the queried unit */
	if (hit_unit)
	{
		deleteLruList(hit_unit);
		insertLruList(hit_unit);
		// current_unit = hit_unit;
		val = hit_unit->val;
		hit_count ++;
		return LRU_SUCCESS;
	}

	/* the cache doesn't store the queried unit */

	/* check if needed to delete the least recent used unit */
	if (cache_count == cache_size)
	{
		// delete the tail unit from the two lists
		hit_unit = lru_tail;
		//deleteBucketList(&cache_bucket[hit_unit->index % cache_size], hit_unit);
		deleteBucketList(&cache_bucket[ValType::hash(hit_unit->index) % bucket_count], hit_unit);
		deleteLruList(hit_unit);
	}
	else
	{
		hit_unit =  new __CacheUnit<ValType>();
		cache_count ++;
	}

	// here hit_unit is the new unit need to inserted 
	hit_unit->index = index;
	//indexedReadFromFile(index, hit_unit->vert.x, hit_unit->vert.y, hit_unit->vert.z);
	indexedReadFromFile(index, hit_unit->val);
	/* write the return value */
	val = hit_unit->val;
	insertBucketList(&cache_bucket[bucket_index], hit_unit);
	insertLruList(hit_unit);
	//current_unit = hit_unit;

#ifdef _WRITE_CACHE_DEBUG
	writeCacheDebug();
#endif

	return LRU_SUCCESS;
}

// insert from head
template <class ValType>
void LRUCache<ValType>::insertBucketList(__CacheUnit<ValType> **head, __CacheUnit<ValType> *new_unit)
{
	new_unit->bucket_prev = NULL;
	new_unit->bucket_next = *head;
	if (*head) {
		(*head)->bucket_prev = new_unit;
	}
	(*head) = new_unit;
}

// delete from any place
template <class ValType>
void LRUCache<ValType>::deleteBucketList(__CacheUnit<ValType> **head, __CacheUnit<ValType> *unit)
{
	if(*head == NULL)
		return;

	if(*head == unit) {
		*head = unit->bucket_next;
	}
	if (unit->bucket_prev) 
		unit->bucket_prev->bucket_next = unit->bucket_next;
	}
	if (unit->bucket_next) {
		unit->bucket_next->bucket_prev = unit->bucket_prev;
	}
}

// insert from the LRU list head
template <class ValType>
void LRUCache<ValType>::insertLruList(__CacheUnit<ValType> *new_unit)
{
	new_unit->use_prev = NULL;
	new_unit->use_next = lru_head;
	if (lru_head) {
		lru_head->use_prev = new_unit;
	}

	if (lru_head == NULL) {
		lru_tail = new_unit;
	}

	lru_head = new_unit;
}

// delete the '_CacheUnit *unit' from the LRU list in any place
template <class ValType>
void LRUCache<ValType>::deleteLruList(__CacheUnit<ValType> *unit)
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

template <class ValType>
int LRUCache<ValType>::indexedReadFromFile(unsigned int index, ValType &val) {

	if (cfile) {
		fsetpos(fp, start_pos_c);
		// !! the range of fseek is less than 4G !!
		fseek(fp, index * ValType::size(), SEEK_CUR);

		if (val.read(fp))
			return LRU_SUCCESS;
		else
			return LRU_FAIL;
	}
	else {
		fin->seekg(start_pos + index * ValType::size());
		fin->seekg(index * ValType::size(), fstream::cur);

		if (val.read(*fin))
			return LRU_SUCCESS;
		else
			return LRU_FAIL;
	}
}

template <class ValType>
int LRUCache<ValType>::closeWriteFile()
{
	fout.close();

	return LRU_SUCCESS;
}

template <class ValType>
int LRUCache<ValType>::closeReadFile()
{
	fin->close();

	return LRU_SUCCESS;
}

template <class ValType>
void LRUCache<ValType>::writeCacheDebug()
{
	int i, c1, c2;
	using namespace std;
	ofstream fout ("cache_dump.txt", ios::out | ios::app);
	__CacheUnit<ValType> *pUnit;

	for(i = 0, c1 = 0; i < bucket_count; i ++)
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

#endif //__LRU_CACHE__