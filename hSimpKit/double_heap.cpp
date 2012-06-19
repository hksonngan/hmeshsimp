#include "double_heap.h"
#include <iostream>

using std::cout;
using std::endl;

template<class T>
doubleHeap<T>::doubleHeap(int _size, int _type)
{
    max_size = DEFAULT_CAPACITY;
	user_set_size = DEFAULT_CAPACITY;
	if(_size >= DEFAULT_CAPACITY) {
        max_size = _size;
		user_set_size = _size;
	}
    data = new T[max_size];
    size = 0;
    type = 1; // default max heap
    if(_type == 1 || type == 2)
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
    if(type == 1) // max heap
    {
        return data[i] < data[j];
    }
    else // min heap
    {
        return data[i] > data[j];
    }
}

template<class T>
void doubleHeap<T>::printHeap(int root, int level)
{
    int i;

    if(root >= size)
        return;

    printHeap(leftChild(root), level + 1);
    for(i = 0; i < level; i ++)
        cout << "\t";
    cout << data[root] << endl;
    printHeap(rightChild(root), level + 1);
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
