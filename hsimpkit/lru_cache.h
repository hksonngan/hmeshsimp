/*
	Manipulations for the LRU cache file.

	There is a least recent used hash cache
	for reading, the hashed value is the index 
	of the value. So the using scenario is that
	there are some values that is too big to 
	fit in the main memory while residing one
	by one in the binary file, and there are
	some reading mechanics that read the value
	based on the location that they are residing 
	in the disk (the index)

	For writing, this is simple binary(!) stream 
	writing.

	The hash bucket with LRU list is like this

	_____________        _____          _____
	|bucket_head| ---->  |   | -> .. -> |   | -> Null
	_____________        _____          _____
	|bucket_head| ---->  |   | -> .. -> |   | -> Null
	_____________        
	|bucket_head| ---->  Null
	
	...
	_____________        _____          _____
	|bucket_head| ---->  |   | -> .. -> |   | -> Null

	While all the nodes in the buckets are linked
	based on the time they are accessed, thus the
	newly accessed node are inserted into the head
	of the LRU list, if it has already existed in
	the list, it will be deleted from the list first

    Author: Ht
    Email : waytofall916@gmail.com
  
    Copyright (C) Ht-waytofall. All rights reserved.
*/

#ifndef __LRU_CACHE__
#define __LRU_CACHE__

#include <fstream>
#include <iostream>
using std::ifstream;
using std::ofstream;


#define LRU_SUCCESS	1
#define LRU_FAIL	0

/* ============================= & DEFINITION & ============================== */

// if dump the cache to the file
//#define _WRITE_CACHE_DEBUG

/* you should inherit this class if you want to
   use the LRU cache, and make this class as the
   template argument for the class template 'LRUCache' */
class LRUVal {
public:
	// binary(!) read
	virtual bool read(ifstream& fin) const = 0;
	// binary(!) write
	virtual bool write(ofstream& fout) const = 0;
	// hash the index
	virtual static unsigned int hash(unsigned int index) const = 0;
	// the size of the 
	virtual static size_t size() = 0;
};

// the equal functor should look like this
/*
class Equal {
public:
	bool operator() (const ValType& val1, const ValType& val2) const {
		return true;
	}
};*/

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
template <
class ValType,	// type of the value contained in the cache, this type must be a derivative of LRUVal
class Equal>	// the equal functor
class LRUCache
{
	/// deprecated
	//typedef struct __Vertex
	//{
	//	float x, y, z;
	//} __Vertex;
	///

public:
	LRUCache();
	~LRUCache();

	/* all the int returned value denotes success
	   unless explicitly stated: 1-success 0-fail */ 

	/* write */
	int openForWrite(const char* filename);
	int closeWriteFile();
	int writeVal(ValType val);

	/* read */
	int initCache(unsigned int size);
	int openForRead(const char* filename);
	// read the value of the given index
	ValType indexedRead(
		unsigned int index,	/*given index*/ 
		ValType &val);		/*returned value*/
	int closeReadFile();

	/* for debug */
	void writeCacheDebug();
	
	/// deprecated
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
	int indexedReadFromFile(unsigned int index, ValType &val);
	
	/// deprecated
	//int indexedReadFromFile(unsigned int index, float &x, float &y, float &z);
	///

private:
	ifstream fin;
	ofstream fout;
	unsigned int cache_size;
	unsigned int cache_count;
	__CacheUnit<ValType> **cache_bucket;
	__CacheUnit<ValType> *lru_head;
	__CacheUnit<ValType> *lru_tail;
	
	/// deprecated
	//__CacheUnit *current_unit;
	///

public:
	unsigned int read_count;
	unsigned int hit_count;
};


/* ============================= & IMPLEMENTATION & ============================== */

template <class ValType, class Equal>
LRUCache<ValType, Equal>::LRUCache()
{
	cache_bucket = NULL;
	cache_size = 0;
	cache_count = 0;
}

template <class ValType, class Equal>
LRUCache<ValType, Equal>::~LRUCache()
{
	if (cache_bucket)
	{
		delete[] cache_bucket;
	}
}

template <class ValType, class Equal>
int LRUCache<ValType, Equal>::openForWrite(const char *filename)
{
	fout.open(filename, ios::out | ios::binary);

	if(fout.good())
		return LRU_SUCCESS;
	else
		return LRU_FAIL;
}

/// deprecated
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

template <class ValType, class Equal>
int LRUCache<ValType, Equal>::writeVal(ValType val) {
	if (val.write(fout)) 
		return LRU_SUCCESS;
	return LRU_FAIL;
}

template <class ValType, class Equal>
int LRUCache<ValType, Equal>::openForRead(const char* filename)
{
	fin.open(filename, ios::binary | ios::in);
	if (fin.good())
		return LRU_SUCCESS;
	else
		return LRU_FAIL;
}

template <class ValType, class Equal>
int LRUCache<ValType, Equal>::indexedRead(unsigned int index, ValType &val)
{
	if (cache_size <= 0)
	{
		cerr << "#error : no cache while indexed reading" << endl;
		return LRU_FAIL;
	}

	read_count ++;

	// the bucket index of the value of the given index
	int bucket_index = ValType::hash(index) % cache_size;
	__CacheUnit<ValType> *hit_unit = NULL;

	// check if the cache hit
	if(cache_bucket[bucket_index])
	{
		__CacheUnit<ValType> *bucket_head;
		for(bucket_head = cache_bucket[bucket_index]; bucket_head; bucket_head = bucket_head->bucket_next) 
		{
			if (bucket_head->index == index) {
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
		// current_unit = hit_unit;
		val = hit_unit->val;
		hit_count ++;
		return LRU_SUCCESS;
	}

	// the cache doesn't store the queried unit

	// check if needed to delete the least recent used unit
	if (cache_count == cache_size)
	{
		// delete the tail unit from the two lists
		hit_unit = lru_tail;
		//deleteBucketList(&cache_bucket[hit_unit->index % cache_size], hit_unit);
		deleteBucketList(&cache_bucket[ValType::hash(hit_unit->index) % cache_size], hit_unit);
		deleteLruList(hit_unit);
	}
	else
	{
		hit_unit =  new __CacheUnit<ValType>();
		cache_count ++;
	}

	// hear hit_unit is the new unit need to inserted 
	hit_unit->index = index;
	//indexedReadFromFile(index, hit_unit->vert.x, hit_unit->vert.y, hit_unit->vert.z);
	indexedReadFromFile(index, hit_unit->val);
	insertBucketList(&cache_bucket[bucket_index], hit_unit);
	insertLruList(hit_unit);
	//current_unit = hit_unit;

#ifdef _WRITE_CACHE_DEBUG
	writeCacheDebug();
#endif

	return LRU_SUCCESS;
}

template <class ValType, class Equal>
int LRUCache<ValType, Equal>::initCache(unsigned int size)
{
	if(size <= 0)
		return 0;

	cache_size = size;
	cache_bucket = new __CacheUnit<ValType>*[cache_size];

	for(int i = 0; i < cache_size; i ++)
		cache_bucket[i] = NULL;

	cache_count = 0;
	lru_head = NULL;
	lru_tail = NULL;

	read_count = 0; 
	hit_count = 0;

	return LRU_SUCCESS;
}

// insert from head
template <class ValType, class Equal>
void LRUCache<ValType, Equal>::insertBucketList(__CacheUnit<ValType> **head, __CacheUnit<ValType> *new_unit)
{
	new_unit->bucket_prev = NULL;
	new_unit->bucket_next = *head;
	if (*head) {
		(*head)->bucket_prev = new_unit;
	}
	(*head) = new_unit;
}

// delete from any place
template <class ValType, class Equal>
void LRUCache<ValType, Equal>::deleteBucketList(__CacheUnit<ValType> **head, __CacheUnit<ValType> *unit)
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
template <class ValType, class Equal>
void LRUCache<ValType, Equal>::insertLruList(__CacheUnit<ValType> *new_unit)
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
template <class ValType, class Equal>
void LRUCache<ValType, Equal>::deleteLruList(__CacheUnit<ValType> *unit)
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

/// deprecated
//template <class ValType, class Equal>
//int LRUCache<ValType, Equal>::indexedReadFromFile(unsigned int index, float &x, float &y, float &z)
//{
//	if (!fin.good())
//	{
//		return 0;
//	}
//
//	fin.seekg(index * 3 * sizeof(float));
//	fin.read((char*)&x, sizeof(float));
//	fin.read((char*)&y, sizeof(float));
//	fin.read((char*)&z, sizeof(float));
//
//	return 1;
//}
///

template <class ValType, class Equal>
int LRUCache<ValType, Equal>::indexedReadFromFile(unsigned int index, ValType &val) {

	fin.seekg(index * ValType::size());
	val.read(fin);

	if (fin.good())
		return LRU_SUCCESS;
	else
		return LRU_FAIL;
}

template <class ValType, class Equal>
int LRUCache<ValType, Equal>::closeWriteFile()
{
	fout.close();

	return LRU_SUCCESS;
}

template <class ValType, class Equal>
int LRUCache<ValType, Equal>::closeReadFile()
{
	fin.close();

	return LRU_SUCCESS;
}

//float LRUCache::getXFloat()
//{
//	return current_unit->vert.x;
//}
//
//float LRUCache::getYFloat()
//{
//	return current_unit->vert.y;
//}
//
//float LRUCache::getZFloat()
//{
//	return current_unit->vert.z;
//}

template <class ValType, class Equal>
void LRUCache::writeCacheDebug()
{
	int i, c1, c2;
	using namespace std;
	ofstream fout ("cache_dump.txt", ios::out | ios::app);
	__CacheUnit<ValType> *pUnit;

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

#endif //__LRU_CACHE__