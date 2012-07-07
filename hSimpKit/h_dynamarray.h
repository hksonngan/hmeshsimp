/*
 *  Dynamic array, enlarged * 2 everytime.
 *  For detailed analysis, please refer to 
 *	amortized analysis chapters in CLRS
 *
 *  Author: Ht
 *  Email:  waytofall916@gmail.com
 */

#ifndef __H_DYNAM_ARRAY__
#define __H_DYNAM_ARRAY__

#include "h_algorithm.h"
#include <iostream>
#include "random.h"

using std::ostream;
using std::endl;

template<class ElemType>
class HDynamArray {
public:
	HDynamArray(int _init_cap = DEFAULT_INIT_CAP);
	~HDynamArray();
	
	/* accessors */
	inline ElemType& operator[] (unsigned int i) const { return data[i]; }
	inline ElemType& elem(unsigned int i) const { return data[i]; }
	inline ElemType* pointer(unsigned int i) { return data + i; }
	inline int count() const { return size; }
	inline int getCapacity() { return capacity; }
	// return the index of the element value 
	// equals to e, return 'size' if it doesn't
	// exist
	inline unsigned int find(ElemType &e);
	inline bool exist(ElemType &e);

	/* modifiers */
	inline void push_back(ElemType e);
	inline void remove(unsigned int index);
	inline void resize(int _capacity);
	void clear() { size = 0; }
	inline void freeSpace();
	void quick_sort() {	h_quick_sort<ElemType, ElemType*>(data, size); }
	// merge with another arr, the capacity
	// is at least size1 + size2
	void merge(HDynamArray<ElemType> &arr2);

	void randGen(unsigned int count, unsigned int range);

private:
	ElemType *data;
	// initial capacity
	int init_cap;
	// capacity
	int capacity;
	// size
	int size;

	static const int DEFAULT_INIT_CAP = 8;
};

template<class ElemType>
HDynamArray<ElemType>::HDynamArray(int _init_cap) {

	if (_init_cap < 0) {
		_init_cap = 0;
	}

	init_cap = _init_cap;
	capacity = init_cap;
	size = 0;
	data = NULL;
	if (capacity != 0) {
		data = new ElemType[capacity];
	}
}

template<class ElemType>
HDynamArray<ElemType>::~HDynamArray() {
	
	freeSpace();
}

template<class ElemType>
void HDynamArray<ElemType>::push_back(ElemType e) {

	if (size >= capacity) {
		if (capacity == 0) {
			capacity = 1;
		}
		resize(capacity * 2);
	}

	data[size] = e;
	size ++;
}

template<class ElemType>
void HDynamArray<ElemType>::resize(int _capacity)  {

	if (_capacity > capacity) {
		ElemType *new_data = new ElemType[_capacity];
		if (capacity != 0)
			memcpy(new_data, data, sizeof(ElemType) * capacity);
		capacity = _capacity;
		if (data)
			delete[] data;
		data = new_data;
	}
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
unsigned int HDynamArray<ElemType>::find(ElemType &e) {
	
	int i;

	for (i = 0; i < size; i ++)
		if (data[i] == e)
			break;

	return i;
}

template<class ElemType>
bool HDynamArray<ElemType>::exist(ElemType &e) {
	
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