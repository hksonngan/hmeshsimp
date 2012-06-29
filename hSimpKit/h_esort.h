/*
 *  multi-way merge external sort
 *
 *  the fundamental knowledge concerning external sort
 *  or even internal sort can be retrieved from books,
 *  like say, TAOCP Volume 3, which is an encyclopedia
 *  for computer science. though most of the contents
 *  are about tape-based external sort, the thoughts
 *  look the same.
 *  have fun.
 *    -ht waytofall916@gmail.com
 */

#ifndef __H_E_SORT__
#define __H_E_SORT__

#include <stdio.h>
#include <list>
#include <string>
#include <iostream>
#include "os_dependent.h"
#include "double_heap.h"

using std::list;
using std::string;
using std::cerr;
using std::endl;


/* ======================== $ INTERNAL PART $ ========================= */

/* internal radixable record interface */
class hRadixRecord {
public:
	// get digit for a particular record.
	// 0 denotes the least significant digit,
	// while getDigitCount() - 1 denotes the
	// most significant one
	virtual int getDigit(int which_digit) const = 0;
	// how many digits 
	virtual int getDigitCount() const = 0;
	// how many variants a digit of the record may contains
	virtual int getDigitSize(int which_digit) const = 0;
};

/* internal radix sort based on listed distribution.
   more details please refer to TAOCP Volume 3 5.2.5.
   the 'ContainerType' should be any random access container
   contains a derivative class of hRadixRecord, overloading
   the accessing operator [] */
template<class ContainerType>
class hIRadixSort {
public:
	hIRadixSort();
	~hIRadixSort();
	bool operator() (ContainerType arr, int arr_count);

private:
	bool bucketSort(ContainerType arr, int arr_count, int which_digit);

private:
	int *index;
	int index_count;
	list<int> *bucket;
	int bucket_count;
};

template<class ContainerType> 
hIRadixSort::hIRadixSort() {

	index = NULL;
	index_count = 0;
	bucket = NULL;
	bucket_count = 0;
}

template<class ContainerType> 
hIRadixSort::~hIRadixSort() {

	if (index) {
		delete[] index;
	}

	if (bucket) {
		delete[] bucket;
	}
}

template<class ContainerType> 
bool hIRadixSort::operator() (ContainerType arr, int arr_count) {

	int i;

	if (index_count < arr_count) {
		if (index) {
			delete[] index;
		}

		index = new int[arr_count];
		index_count = arr_count;
	}

	for (i = 0; i < arr_count; i ++) {
		index[i] = i;
	}

	for (i = 0; i < arr[0].getDigitCount(); i ++) {
		if (bucketSort(arr, arr_count, i) == false)
			return false;
	}

	return true;
}

template<class ContainerType> 
bool hIRadixSort::bucketSort(ContainerType arr, int arr_count, int which_digit) {

	int digit_size = arr[0].getDigitSize(which_digit);

	if (bucket_count < digit_size) {
		if (bucket) {
			delete[] bucket;
		}

		bucket = new list<int>[digit_size];
		bucket_count = digit_size;
	}

	hRadixRecord *record;
	int i;

	for (i = 0; i < arr_count; i ++) {
		record = &arr[index[i]];
		bucket[record->getDigit(which_digit)].push_back(index[i]);
	}

	list<int>::iterator iter;
	int j;
	for (i = 0, j = 0; i < digit_size; i ++) {
		for (iter = bucket[i].begin(); iter != bucket[i].end(); iter ++, j ++) 
			index[j] = *iter;
	}

	return true;
}


/* ======================== $ EXTERNAL PART $ ========================= */

/* external record interface */
class hERecord {
public:
	/* read a record from file */
	virtual bool read(FILE *fp) = 0;
	/* write a record to file */
	virtual bool write(FILE *fp) = 0;
	virtual bool operator <(const hERecord &r) const = 0;
	virtual bool operator >(const hERecord &r) const = 0;
};

/* external radixable record interface */
class hERadixRecord : public hERecord, public hRadixRecord {};

/* external sort common operations 
   'ERadixRecordType' must be a derivative of hERadixRecord
*/
template<class ERadixRecordType>
class hESortCommon {

	/* a class acting as the value stored in the heap
	   the 'index' acts as a index to the container of
	   ERadixRecordType, 'container' acts as a container
	   which contains one just-loaded records from every
	   patches */
	template<class ERadixRecordType>
	class HeapableIndex {
	public:
		bool operator <(const HeapableIndex &hi) {
			return container[index] < container[hi.index];
		}

		bool operator >(const HeapableIndex &hi) {
			return container[index] > container[hi.index];
		}

		void set(int _index, ERadixRecordType* _container) {
			index = _index;
			container = _container;
		}

	public:
		int index;
		ERadixRecordType* container;
	};

public:
	bool set(
		FILE* _fin,
		char* _infilename,
		int _records_in_a_patch, // count of records in a file patch
		int _record_count = 0 // equals or less than 0 indicates that the terminate sign is end of file 
	);
	/*
	 * return value: if the functions has ran right */
	bool nextPatch(ERadixRecordType* arr, int &read_count, int &patch_id, bool &over);
	bool writePatch(ERadixRecordType* arr, int arr_count, int patch_id);
	bool merge(char *outfilename);

private:
	FILE* fin;
	char* infilename;
	int records_in_a_patch; // count of records in a file patch
	int record_count; // equals or less than 0 indicates that the terminate sign is end of file 
	int patches_count;
	char name_buf[500];
	string dir_name;
};

template<class ERadixRecordType>
bool hESortCommon::set (
	FILE* _fin,
	char* _infilename,
	int _records_in_a_patch, // count of records in a file patch
	int _record_count = 0 // equals or less than 0 indicates that the terminate sign is end of file 
	) {

	fin = _fin;
	infilename = _infilename;
	records_in_a_patch = _records_in_a_patch;
	record_count = _record_count;
	patches_count = 0;

	// create a directory for holding the patches
	dir_name = getFilename(infilename);
	dir_name += "_patches";
	stringToCstr(dir_name, name_buf);
	if (hCreateDir(name_buf) == false) {
		cerr << "#hESortCommon::set(): can't create directory" << endl;
		return false;
	}
	
	return true;
}

template<class ERadixRecordType>
bool hESortCommon::writePatch(ERadixRecordType* arr, int arr_count, int patch_id) {

	char buf[40];
	sprintf(buf, "%d", patch_id);
	string patch_name = dir_name + hPathSeperator() + buf;
	FILE *fp = fopen(patch_name, "wb");
	int i;

	if (fp == NULL) {
		cerr << "#hESortCommon::writePatch(): error open file for writing" << endl
			<< "patch id: " << patch_id;
	}

	for (i = 0; i < arr_count; i ++) {
		if (arr[i].write(fp) == false) {
			cerr << "#hESortCommon::writePatch(): error writing records" << endl;
			return false;
		}
	}

	fclose(fp);
	return true;
}

template<class ERadixRecordType>
bool hESortCommon::nextPatch(ERadixRecordType* arr, int &read_count, int &patch_id, bool &over) {

	over = false;

	for (read_count = 0; read_count < records_in_a_patch; read_count ++) {
		if (record_count > 0) {
			if (patches_count * records_in_a_patch + read_count >= record_count) {
				over = true;
				break;
			}
		}

		if (arr[read_count].read(fin) == false) {
			if (feof(fin)) {
				over = true;
				break;
			}
			else if (ferror(fin)) {
				cerr << "#hESortCommon::nextPatch(): error occurred when reading file" << endl;
				return false;
			}
		}
	}

	patch_id = patches_count;
	patches_count ++;

	return true;
}

template<class ERadixRecordType>
bool hESortCommon::merge(char *outfilename) {

	DoubleHeap<HeapableIndex<ERadixRecordType>> heap(patches_count, MinHeap);
	ERadixRecordType *container = new ERadixRecordType[patches_count];
	int i;

	for (i = 0; i < patches_count; i ++) {
	}
}

/* external radix sort
   'ERadixRecordType' must be a derivative of hERadixRecord */
template<class ERadixRecordType>
bool hERadixSort (
	FILE* fin,
	char* infilename,
	char* outfilename,
	int records_in_a_patch, // count of records in a file patch
	int record_count = 0 // equals or less than 0 indicates that the terminate sign is end of file 
	){

	int read_count, patch_id, num;
	ERadixRecordType *internal_records;
	bool over = false;
	hIRadixSort<ERadixRecordType*> iRadixSort;
	hESortCommon<ERadixRecordType> eSortCommon;

	internal_records = new ERadixRecordType[records_in_a_patch];
	eSortCommon.set(fin, infilename, records_in_a_patch, record_count);
	
	while (!over) {

		if (eSortCommon.nextPatch(internal_records, read_count, patch_id, over) == false) {
			return false;
		}

		iRadixSort(internal_records, read_count);

		if (eSortCommon.writePatch(internal_records, read_count, patch_id) == false) {
			return false
		}
	}

	eSortCommon.merge(outfilename);

	delete[] internal_records;
	return true;
}

#endif //__H_E_SORT__