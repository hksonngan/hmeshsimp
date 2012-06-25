/*
 *  dynamic array, enlarged * 2 everytime
 */

#ifndef __H_DYNAM_ARRAY__
#define __H_DYNAM_ARRAY__

template<class ElemType>
class HDynamArray {
public:
	HDynamArray(int _init_cap = DEFAULT_INIT_CAP);
	~HDynamArray();
	void push_back(ElemType e);
	ElemType& operator[] (int i);
	int count();

private:
	ElemType *data;
	// initial capacity
	int init_cap;
	// capacity
	int capacity;
	// size
	int size;

	static int DEFAULT_INIT_CAP = 8;
};

template<class ElemType>
HDynamArray::HDynamArray(int _init_cap)
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
HDynamArray::~HDynamArray()
{
	delete[] data;
}

template<class ElemType>
void HDynamArray::push_back(ElemType e)
{
	if (size >= capacity) {
		ElemType *new_data = new ElemType[capacity * 2];
		memcpy(new_data, data, size(ElemType) * capacity);
		delete[] data;
		data = new_data;
	}

	data[size] = e;
	size ++;
}

template<class ElemType>
ElemType& HDynamArray::operator[] (int i)
{
	if (i >= 0 && i < size)
		return data[i];
	else
		return ElemType();
}

template<class ElemType>
int HDynamArray::count()
{
	return size;
}

#endif //__H_DYNAM_ARRAY__