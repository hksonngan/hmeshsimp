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
#include "os_dependent.h"

using std::list;
using std::string;


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
		record = arr[index[i]];
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
	virtual read(FILE *fp) = 0;
};

/* external radixable record interface */
class hERadixRecord : public hERecord, public hRadixRecord {};

/* external sort common operations */
bool hESortCommon {

};

/* external radix sort
   'ERadixRecordType' must be a derivative of hERadixRecord */
template<class ERadixRecordType>
bool hERadixSort 
(
	char* infilename,
	char* outfilename,
	int records_in_a_patch, // count of records in a file patch
	int record_count = 0 // equals or less than 0 indicates that the terminate sign is end of file 
){

	FILE* fin;
	int patches_count, i, j;
	FILE* fout;
	char name_buf[500];
	ERadixRecordType *internal_records;
	string dir_name;

	internal_records = new ERadixRecordType[records_in_a_patch];
	dir_name = getFilename(infilename);
	dir_name += "_patches";
	memcpy(name_buf, dir_name.c_str(), dir_name.size() * sizeof(char));
	name_buf[dir_name.size()] = '\0';
    
	for (patches_count = 0; true; patches_count ++) {
		for (j = 0; j < patches_count; j ++) {
			
		}
    }
}
	
	

#endif //__H_E_SORT__