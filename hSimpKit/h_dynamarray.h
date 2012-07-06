/*
 *  dynamic array, enlarged * 2 everytime.
 *  for detailed analysis, please refer to 
 *	amortized analysis chapters in CLRS
 *
 *  author: ht
 *  email:  waytofall916@gmail.com
 */

#ifndef __H_DYNAM_ARRAY__
#define __H_DYNAM_ARRAY__

template<class ElemType>
class HDynamArray {
public:
	HDynamArray(int _init_cap = DEFAULT_INIT_CAP);
	~HDynamArray();
	
	/* accessors */
	ElemType& operator[] (int i) { return data[i]; }
	ElemType& elem(int i) { return data[i]; }
	int count() { return size; }
	int getCapacity() { return capacity; }
	// return the index of the element value 
	// equals to e, return 'size' if it doesn't
	// exist
	inline uint find(ElemType &e);
	inline bool exist(ElemType &e);

	/* modifiers */
	inline void push_back(ElemType e);
	inline void remove(uint &index);
	void resize(int _capacity);
	void clear() { size = 0; }
	// merge with another arr, the capacity
	// is at least size1 + size2
	void merge(HDynamArray &arr2);
	void qsort();

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
HDynamArray<ElemType>::HDynamArray(int _init_cap)
{
	if (_init_cap <= 0) {
		_init_cap = DEFAULT_INIT_CAP;
	}

	init_cap = _init_cap;
	capacity = init_cap;
	size = 0;
	data = new ElemType[capacity];
}

template<class ElemType>
HDynamArray<ElemType>::~HDynamArray()
{
	delete[] data;
}

template<class ElemType>
void HDynamArray<ElemType>::push_back(ElemType e)
{
	if (size >= capacity) {
		resize(capacity * 2);
	}

	data[size] = e;
	size ++;
}

template<class ElemType>
void HDynamArray<ElemType>::resize(int _capacity) 
{
	if (_capacity > capacity) {
		ElemType *new_data = new ElemType[_capacity];
		memcpy(new_data, data, sizeof(ElemType) * capacity);
		capacity = _capacity;
		delete[] data;
		data = new_data;
	}
}

template<class ElemType>
uint HDynamArray<ElemType>::find(ElemType &e) {
	
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
void HDynamArray<ElemType>::remove(uint &index) {

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
void HDynamArray<ElemType>::merge(HDynamArray &arr2) {


}

#endif //__H_DYNAM_ARRAY__