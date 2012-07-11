/*
 *  Multi-way Merge External Sort
 *
 *  The fundamental knowledge concerning external sort
 *  or even internal sort can be retrieved from books,
 *  like say, TAOCP Volume 3, which is an encyclopedia
 *  for computer science. Though most of the contents
 *  are about tape-based external sort, the thoughts
 *  look the same.
 *  have fun.
 *    -Ht waytofall916@gmail.com
 */

#ifndef __H_E_SORT__
#define __H_E_SORT__

#include <stdio.h>
#include <list>
#include <string>
#include <iostream>
#include "os_dependent.h"
#include "double_heap.h"
#include "trivial.h"

using std::list;
using std::string;
using std::cerr;
using std::endl;

#define FILE_NAME_BUF_SIZE 500


/* ======================== $ CLASS & FUNCTUION DEFINED $ ========================= */

class hRadixRecord;
template<class ContainerType> class hIRadixSort;
class hERecord;
/* this class should be inherited. 
   the class that inherits this as
   record class must overload the
   operator '<' and '>' */
class hERadixRecord;
template<class ERadixRecordType> class HeapableIndex;
template<class ERadixRecordType> class hESortCommon;
template<class ERadixRecordType>
bool hERadixSort (
	FILE*	fin,
	char*	infilename,
	char*	outfilename,
	int		records_in_a_patch, // count of records in a file patch
	int		record_count = 0 ); // equals or less than 0 indicates that the terminate sign is end of file 


/* ======================== $ CLASS & TEMPLATE DEFINITION $ ========================= */

/* internal radixable record interface */
class hRadixRecord {
public:
	// get digit for a particular record.
	// 0 denotes the least significant digit,
	// while getDigitCount() - 1 denotes the
	// most significant one
	virtual unsigned int getDigit(int which_digit) const = 0;
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
	/* sort result is the index array, the input array remains the same */
	bool operator() (ContainerType arr, int arr_count);
	int* getIndex() { return index; }

private:
	bool bucketSort(
		ContainerType arr, 
		int arr_count, 
		int which_digit);

private:
	int		*index;
	int		index_count;
	list<unsigned int>	*bucket;
	int		bucket_count;
};


/* external record interface */
class hERecord {
public:
	/* read a record from file, return false if file reading fails */
	virtual bool read(FILE *fp) = 0;
	/* write a record to file */
	virtual bool write(FILE *fp) = 0;
};

/* external radixable record interface */
class hERadixRecord : public hERecord, public hRadixRecord {};

/* a class acting as the value stored in the heap
	   the 'index' acts as a index to the container of
	   ERadixRecordType, 'container' acts as a container
	   which contains one just-loaded records from every
	   patches */
template<class ERadixRecordType>
class HeapableIndex {
public:
	bool operator <(const HeapableIndex &hi) { return container[index] < container[hi.index]; }
	bool operator >(const HeapableIndex &hi) { return container[index] > container[hi.index]; }
	void set(int _index) { index = _index; }

public:
	int index;
	static ERadixRecordType* container;
};

template<class ERadixRecordType>
ERadixRecordType* HeapableIndex<ERadixRecordType>::container;

/* external sort common operations 
   'ERadixRecordType' must be a derivative of hERadixRecord */
template<class ERadixRecordType>
class hESortCommon {
public:
	bool set(
		FILE*	_fin,
		char*	_infilename,
		int		_records_in_a_patch, // count of records in a file patch
		int		_record_count = 0 ); // equals or less than 0 indicates that the terminate sign is end of file 

	/*
	 * return value: if the functions has ran right 
	 */
	bool nextPatch(
		ERadixRecordType* arr, 
		int &read_count, 
		int &patch_id, 
		bool &over);
	bool writePatch(
		ERadixRecordType* arr, 
		int* index, 
		int arr_count, 
		int patch_id);
	bool merge(char *outfilename);

private:
	FILE*	fin;
	char*	infilename;
	int		records_in_a_patch;	// count of records in a file patch
	int		record_count;		// equals or less than 0 indicates that the terminate sign is end of file 
	int		patches_count;
	char	name_buf[500];
	string	dir_name;
};
	

/* ======================== $ IMPLEMENTATION $ ========================= */

/* hIRadixSort */
template<class ContainerType> 
hIRadixSort<ContainerType>::hIRadixSort() {

	index = NULL;
	index_count = 0;
	bucket = NULL;
	bucket_count = 0;
}

template<class ContainerType> 
hIRadixSort<ContainerType>::~hIRadixSort() {

	if (index) {
		delete[] index;
	}

	if (bucket) {
		delete[] bucket;
	}
}

template<class ContainerType> 
bool hIRadixSort<ContainerType>::operator() (ContainerType arr, int arr_count) {

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
bool hIRadixSort<ContainerType>::bucketSort(
	ContainerType arr, 
	int arr_count, 
	int which_digit) {

	int digit_size = arr[0].getDigitSize(which_digit);

	if (bucket_count < digit_size) {
		if (bucket) {
			delete[] bucket;
		}

		bucket = new list<unsigned int>[digit_size];
		bucket_count = digit_size;
	}

	hRadixRecord *record;
	int i;

	for (i = 0; i < arr_count; i ++) {
		record = &arr[index[i]];
		bucket[record->getDigit(which_digit)].push_back(index[i]);
	}

	list<unsigned int>::iterator iter;
	int j;
	for (i = 0, j = 0; i < digit_size; i ++) {
		for (iter = bucket[i].begin(); iter != bucket[i].end(); iter ++, j ++) 
			index[j] = *iter;

		bucket[i].clear();
	}

	return true;
}

/* hESortCommon */
template<class ERadixRecordType>
bool hESortCommon<ERadixRecordType>::set (
	FILE*	_fin,
	char*	_infilename,
	int		_records_in_a_patch,	// count of records in a file patch
	int		_record_count = 0 ) {	// equals or less than 0 indicates that the terminate sign is end of file 

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
bool hESortCommon<ERadixRecordType>::nextPatch(
	ERadixRecordType* arr, 
	int &read_count, 
	int &patch_id, 
	bool &over) {

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
bool hESortCommon<ERadixRecordType>::writePatch(
	ERadixRecordType* arr, 
	int* index, 
	int arr_count, 
	int patch_id) {

	// get patch name
	char buf[FILE_NAME_BUF_SIZE];
	sprintf(buf, "%d", patch_id);
	string patch_name = dir_name + hPathSeperator() + buf;
	stringToCstr(patch_name, buf);

	FILE *fp = fopen(buf, "wb");
	int i;

	if (fp == NULL) {
		cerr << "#hESortCommon::writePatch(): error open file for writing" << endl
			<< "patch id: " << patch_id << endl;
	}

	for (i = 0; i < arr_count; i ++) {
		if (arr[index[i]].write(fp) == false) {
			cerr << "#hESortCommon::writePatch(): error writing records" << endl;
			return false;
		}
	}

	fclose(fp);
	return true;
}

template<class ERadixRecordType>
bool hESortCommon<ERadixRecordType>::merge(char *outfilename) {

	DoubleHeap<HeapableIndex<ERadixRecordType>> heap(patches_count, MinHeap);
	ERadixRecordType *container = new ERadixRecordType[patches_count];
	FILE **fps = new FILE*[patches_count], *fout;
	char buf[FILE_NAME_BUF_SIZE];
	string patch_name;
	HeapableIndex<ERadixRecordType> hi;
	int i;

	fout = fopen(outfilename, "wb");
	if (fout == NULL) {
		cerr << "#hESortCommon::merge(): open output file name failed" << endl;
		return false;
	}

	HeapableIndex<ERadixRecordType>::container = container;
	for (i = 0; i < patches_count; i ++) {
		sprintf(buf, "%d", i);
		patch_name = dir_name + hPathSeperator() + buf;
		stringToCstr(patch_name, buf);
		fps[i] = fopen(buf, "rb");
		if (fps[i] == NULL) {
			cerr << "hESortCommon::merge(): open patch file " + i << " failed" << endl;
			return false;
		}

		container[i].read(fps[i]);
		hi.set(i);
		heap.addElement(hi);
	}

	while (!heap.empty()) {
		hi = heap.getTop();
		heap.deleteTop();
		if (container[hi.index].write(fout) == false) {
			cerr << "hESortCommon::merge(): output file writing error" << endl;
			return false;
		}

		if (container[hi.index].read(fps[hi.index])) {
			heap.addElement(hi);
		}
		else {
			if (ferror(fps[hi.index])) {
				cerr << "hESortCommon::merge(): patch file " << hi.index << " reading error" << endl;
				return false;
			}
		}
	}

	// close files
	for (i = 0; i < patches_count; i ++) {
		fclose(fps[i]);
	}
	fclose(fout);

	delete[] container;
	return true;
}

/* hERadixSort */
template<class ERadixRecordType>
bool hERadixSort<ERadixRecordType> (
	FILE*	fin,
	char*	infilename,
	char*	outfilename,
	int		records_in_a_patch,	// count of records in a file patch
	int		record_count ) {	// equals or less than 0 indicates that the terminate sign is end of file 

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

		if (eSortCommon.writePatch(internal_records, iRadixSort.getIndex(), read_count, patch_id) == false) {
			return false;
		}
	}

	eSortCommon.merge(outfilename);

	delete[] internal_records;
	return true;
}

#endif //__H_E_SORT__