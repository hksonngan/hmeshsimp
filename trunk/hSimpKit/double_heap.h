/*
 *  a maximum/minimum heap with an array as a container
 *
 *  constructor:
 *    doubleHeap(
 *      int _size, // initial volume of the heap
 *      int _type // 1 : max heap 2 : min heap
 *    );
 *
 *  the element type T should overload
 *  operator '<' for max heap,
 *           '>' for min heap
 *
 *  author : ht
 *  email  : waytofall916@gmail.com
 */

#ifndef __DOUBLE_HEAP__
#define __DOUBLE_HEAP__

#include <ostream>

using std::ostream;
using std::endl;

enum HeapType {
	MaxHeap, MinHeap
};

template<class T>
class doubleHeap
{
public:
	doubleHeap(
		int _size,
		HeapType _type
		);
	~doubleHeap() { delete[] data; }
	void getTop(T &top);
	T* getTopPointer() { return data; };
	T getTop() { return data[0]; };
	// delete the top element and return the content of the top
	T deleteTop();
	// add a new element whose content is a copy of _new_element
	bool addElement(T _new_element);
	bool empty();
	bool full();
	/* int root  : index of the root
	 * int level : count of the level 
	 */
	void printHeap(ostream& out, int root, int level); // for debug
	int count() { return size; }
	T get(int i) { return data[i]; }
	// clear the content while half the capacity if it's too big
	void clear();

private:
	void swap(int i, int j);
	int leftChild(int i);
	int rightChild(int i);
	int parent(int i);
	bool comp(int i, int j); // if the nodes should exchange

private:
	T *data;
	int max_size;
	int user_set_size;
	int size;
	HeapType type;
	static const int DEFAULT_CAPACITY = 100;
};

#include "double_heap.h"
#include <iostream>

using std::cout;
using std::endl;

template<class T>
doubleHeap<T>::doubleHeap(int _size, HeapType _type = MaxHeap) // default max heap
{
	max_size = DEFAULT_CAPACITY;
	user_set_size = DEFAULT_CAPACITY;
	if(_size >= DEFAULT_CAPACITY) {
		max_size = _size;
		user_set_size = _size;
	}
	data = new T[max_size];
	size = 0;
	type = _type;
}

template<class T>
void doubleHeap<T>::getTop(T &top)
{
	if(size == 0)
		return;

	top = data[0];
}

template<class T>
T doubleHeap<T>::deleteTop()
{
	T deleted;

	if(size == 0)
		return deleted;

	deleted = data[0];
	data[0] = data[size - 1];
	size --;

	int cur = 0; // start from the root
	int lChildIndex;
	int rChildIndex;

	// begin exchanging the node and check if it's been a heap
	while(true)
	{
		if(cur >= size) // the heap is null
			break;

		rChildIndex = rightChild(cur);
		lChildIndex = leftChild(cur);

		if(lChildIndex >= size) // right child and left child has been a null
			break;
		else if(rChildIndex >= size) // rightChild null, left not
		{
			if(comp(cur, lChildIndex))
			{
				swap(cur, lChildIndex);
				cur = lChildIndex;
			}
			else
				break; // has been a heap
		}
		else // left and right are not null
		{
			if(comp(cur, rChildIndex) || comp(cur, lChildIndex))
			{
				if(comp(lChildIndex, rChildIndex))
				{
					swap(cur, rChildIndex);
					cur = rChildIndex;
				}
				else
				{
					swap(cur, lChildIndex);
					cur = lChildIndex;
				}
			}
			else
				break;
		}
	}

	return deleted;
}

template<class T>
bool doubleHeap<T>::addElement(T _new_element)
{
	// realloc the data space
	if (size + 1 >= max_size)
	{
		T *new_data = new T[max_size * 2];
		memcpy(new_data, data, sizeof(T) * max_size);
		delete[] data;
		data = new_data;
		max_size *= 2;
	}

	// value assignment
	data[size] = _new_element;
	size ++;

	int cur = size - 1;
	int parentIndex;
	while(true)
	{
		if(cur == 0)
			break;

		parentIndex = parent(cur);
		if(comp(parentIndex, cur))
		{
			swap(cur, parentIndex);
			cur = parentIndex;
		}
		else
			break;
	}

	return true;
}

template<class T>
bool doubleHeap<T>::empty()
{
	return size == 0;
}

template<class T>
bool doubleHeap<T>::full()
{
	return max_size == size;
}

template<class T>
void doubleHeap<T>::clear()
{
	size = 0;

	if (max_size > user_set_size) {
		delete[] data;
		data = new T[max_size / 2];
		max_size /= 2;
	}
}

template<class T>
void doubleHeap<T>::swap(int i, int j)
{
	T ex;
	ex = data[i];
	data[i] = data[j];
	data[j] = ex;
}

template<class T>
int doubleHeap<T>::leftChild(int i)
{
	return 2 * (i + 1) - 1;
}

template<class T>
int doubleHeap<T>::rightChild(int i)
{
	return 2 * (i + 1);
}

template<class T>
int doubleHeap<T>::parent(int i)
{
	return (i + i) / 2 - 1;
}

template<class T>
bool doubleHeap<T>::comp(int i, int j)
{
	if(type == MaxHeap) // max heap
	{
		return data[i] < data[j];
	}
	else // min heap
	{
		return data[i] > data[j];
	}
}

template<class T>
void doubleHeap<T>::printHeap(ostream& out, int root, int level)
{
	int i;

	if(root >= size)
		return;

	printHeap(out, leftChild(root), level + 1);
	for(i = 0; i < level; i ++)
		out << "\t";
	out << data[root] << endl;
	printHeap(out, rightChild(root), level + 1);
}

/* helper functions */

template<class T>
inline void PrintHeap(doubleHeap<T> &h)
{
#ifdef PRINT_HEAP
	if (h.empty())
		return;

	h.printHeap(cout, 0, 0);
	cout << endl << endl;
#endif
}

//int main()
//{
//    int a[] = {1, 10, 6, 23, 7, 8, 90, 12, 45, 76, 33, 25, 3, 17, 70, 10};
//    int i, aLen = 16, e;
//    doubleHeap<int> maxHeap(100, 1);
//
//    for(i = 0; i < aLen; i ++)
//    {
//        maxHeap.addElement(a[i]);
//    }
//
//    maxHeap.printHeap(0, 0);
//
//    // heap sort
//    while(!maxHeap.empty())
//    {
//        maxHeap.extractTop(e);
//        cout << e << " ";
//        maxHeap.deleteTop();
//    }
//
//    return 0;
//}

#endif //__DOUBLE_HEAP__