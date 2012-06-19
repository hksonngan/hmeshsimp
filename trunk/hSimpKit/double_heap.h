/*
 *  a maximum/minimum heap with an array as a container
 *  constructor:
 *    doubleHeap(
 *      int _size, // initial volume of the heap
 *      int _type // 1 : max heap 2 : min heap
 *    );
 *  the element type T should overload
 *  operator '<' for max heap,
 *  '>' for min heap
 *
 *  author : ht
 *  email  : waytofall916@gmail.com
 */

#ifndef __DOUBLE_HEAP__
#define __DOUBLE_HEAP__

template<class T>
class doubleHeap
{
public:
	doubleHeap(
		int _size,
		int _type // 1 : max heap 2 : min heap
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
	void printHeap(int root, int level); // for debug
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
	int type;
	static const int DEFAULT_CAPACITY = 100;
};

#endif //__DOUBLE_HEAP__