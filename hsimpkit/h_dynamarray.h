/*
 *  Dynamic array, enlarged * 2 everytime.
 *
 *  Author: Ht
 *  Email : waytofall916 at gmail dot com
 *
 *  Copyright (C) Ht. All rights reserved.
 */

#ifndef __H_DYNAM_ARRAY__
#define __H_DYNAM_ARRAY__

#include "h_algorithm.h"
#include <iostream>
#include "random.h"

using std::ostream;
using std::endl;

// Dynamic array, enlarged * 2 everytime.
template<class ElemType>
class HDynamArray {
public:
	HDynamArray(char* _id = NULL, int _init_cap = 0);
	~HDynamArray();
	
	/* accessors */
	inline ElemType& operator[] (unsigned int i) const { return data[i]; }
	inline ElemType& elem(unsigned int i) const { return data[i]; }
	inline ElemType& at(unsigned int i) const { return data[i]; }
	inline ElemType* pointer(unsigned int i) { return data + i; }
	inline int count() const { return size; }
	inline int getCapacity() const { return capacity; }
	// return the index of the element value 
	// equals to e, return 'size' if it doesn't
	// exist
	inline unsigned int find(ElemType &e) const;
	inline bool exist(ElemType &e) const;

	/* modifiers */
	inline void push_back(ElemType e);
	inline void remove(unsigned int index);
	inline void resize(int _capacity);
	// this may cause the array to be trimmed
	inline void setCount(int _new_count);
	void clear() { size = 0; }
	inline void freeSpace();
	void quick_sort() {	h_quick_sort<ElemType, ElemType*>(data, size); }
	// merge with another arr, the capacity
	// is at least size1 + size2
	void merge(HDynamArray<ElemType> &arr2);
	// CAUTION!!: all the value assign is a shallow
	// copy which may cause a 'data' field deleted
	// when the argument is destructed, thus causing
	// problems. so when using HDynamArray as a parameter,
	// DO USE REFERENCES OR POINTERS.
	void swap(HDynamArray<ElemType> &arr2);
	// !!this may cause memory leakage
	void setNULL();

	void randGen(unsigned int count, unsigned int range);

protected:
	ElemType *data;
	// initial capacity
	int init_cap;
	// capacity
	int capacity;
	// size
	int size;
	// name for debug
	char ID[100];

	static const int DEFAULT_INIT_CAP = 8;
};

template<class ElemType>
HDynamArray<ElemType>::HDynamArray(char *_id, int _init_cap) {
	if (_init_cap <= 0) {
		init_cap = DEFAULT_INIT_CAP;
		capacity = 0;
	}
	else {
		init_cap = _init_cap;
		capacity = _init_cap;
	}

	size = 0;
	data = NULL;
	if (capacity > 0) {
		data = new ElemType[capacity];
	}

	if (_id)
		memcpy(ID, _id, strlen(_id) + 1);
}

template<class ElemType>
HDynamArray<ElemType>::~HDynamArray() {
	freeSpace();
}

template<class ElemType>
void HDynamArray<ElemType>::push_back(ElemType e) {
	if (size >= capacity) {
		if (capacity == 0) 
			resize(init_cap);
		else
			resize(capacity * 2);
	}

	data[size] = e;
	size ++;
}

template<class ElemType>
void HDynamArray<ElemType>::resize(int _capacity)  {
	if (_capacity > capacity) {
		ElemType *new_data = new ElemType[_capacity];
		if (data) {
			memcpy(new_data, data, sizeof(ElemType) * capacity);
			delete[] data;
		}
		capacity = _capacity;
		data = new_data;
	}
}

template<class ElemType>
void HDynamArray<ElemType>::setCount(int _new_count) {

	if (_new_count > capacity)
		resize(_new_count);
	
	size = _new_count;
}

template<class ElemType>
void HDynamArray<ElemType>::freeSpace() {

	size = 0;
	capacity = 0;
	if (data) {
		delete[] data;
		data = NULL;
	}
}

template<class ElemType>
unsigned int HDynamArray<ElemType>::find(ElemType &e) const {
	
	int i;

	for (i = 0; i < size; i ++)
		if (data[i] == e)
			break;

	return i;
}

template<class ElemType>
bool HDynamArray<ElemType>::exist(ElemType &e) const {
	
	int i = find(e);

	return i < size;
}

template<class ElemType>
void HDynamArray<ElemType>::remove(unsigned int index) {

	if (size <= 0)
		return;

	if (index >= size)
		return;

	int i;

	for (i = index + 1; i < size; i --) {
		data[i - 1] = data[i];
	}

	size --;
}

template<class ElemType>
void HDynamArray<ElemType>::merge(HDynamArray<ElemType> &arr2) {

	this->quick_sort();
	arr2.quick_sort();

	int new_cap = size + arr2.size;
	if (new_cap < capacity)
		new_cap = capacity;
	ElemType *new_data = new ElemType[new_cap];

	merge_sorted_arr<ElemType, ElemType*>(data, size, arr2.data, arr2.size, new_data);
	capacity = new_cap;
	size += arr2.size;
	delete[] data;
	data = new_data;
}

template<class ElemType>
void HDynamArray<ElemType>::swap(HDynamArray<ElemType> &arr2) {

	ElemType *_data = data;
	int _init_cap = init_cap;
	int _capacity = capacity;
	int _size = size;

	data = arr2.data;
	init_cap = arr2.init_cap;
	capacity = arr2.capacity;
	size = arr2.size;

	arr2.data = _data;
	arr2.init_cap = _init_cap;
	arr2.capacity = _capacity;
	arr2.size = _size;
}

template<class ElemType>
void HDynamArray<ElemType>::setNULL() {

	size = 0;
	capacity = 0;
	data = NULL;
}

template<class ElemType>
void HDynamArray<ElemType>::randGen(unsigned int count, unsigned int range) {

	clear();
	resize(count);

	usrand(0);
	for (int i = 0; i < count; i ++) {
		push_back(urand() % range);
	}
}

///////////////////////////////////////////////////////////////////////////
////

template<class ElemType>
ostream& operator <<(ostream& out, const HDynamArray<ElemType> &arr) {
	
	for (int i = 0; i < arr.count(); i ++)
		out << arr[i] << endl;
	out << endl;

	return out;
}

template<class ElemType>
void merge_dynam_arr(HDynamArray<ElemType> &arr1, HDynamArray<ElemType> &arr2, HDynamArray<ElemType> &dst) {

	dst.resize(arr1.count() + arr2.count());
	merge_arr<ElemType, HDynamArray<ElemType>>(arr1, arr1.count(), arr2, arr2.count(), dst);
}

#endif //__H_DYNAM_ARRAY__