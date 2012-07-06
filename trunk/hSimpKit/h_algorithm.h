/*
 *  algorithm
 *
 *  ht
 *  waytofall916@gmail.com
 */

#ifndef __H_ALGORITHM__
#define __H_ALGORITHM__

#include "random.h"

enum SortType { ASCEND, DESCEND };

/* ---------- array partitioning ---------- */

/* if the element is part of a concerning patch */
template<class ElemType>
class ElemPartOf {
public:
	virtual bool operator() (ElemType e) = 0;
};

/* evoked when swapping two elements */
class NotifySwap {
public:
	virtual void operator() (int i, int j) = 0;
};

/* array self partition like what's in quicksort 
 * able to partition into arbitrary patches */
template<class ElemType, class ContainerType>
class ArraySelfPartition {
public:
	ArraySelfPartition(int _cap = 2);
	~ArraySelfPartition();
	int* operator() (ContainerType &arr, int arr_start, int arr_end, 
		ElemPartOf<ElemType> **part_of, int partion_count, NotifySwap* notify_swap = NULL);

private:
	int *index;
	int indexCapacity;
};

template<class ElemType, class ContainerType>
ArraySelfPartition<ElemType, ContainerType>::ArraySelfPartition(int _cap) {
	if (_cap < 2) {
		_cap = 2;
	}

	index = new int[_cap];
	indexCapacity = _cap;
}

template<class ElemType, class ContainerType>
ArraySelfPartition<ElemType, ContainerType>::~ArraySelfPartition() {
	delete[] index;
}

/* assuming there are 8 partitions, index[0] ~ index[7]
 * denotes the index of the cutting for every partition.
 * so that: 
 *                0 ~ index[0] denotes the 1st partition,
 *     index[0] + 1 ~ index[1] denotes the 2nd partition,
 *     index[1] + 1 ~ index[2] denotes the 3rd partition,
 *     ...
 *     index[6] + 1 ~ index[7] denotes the 8th partition
 *
 * and if index[j - 1] == index[j] (or -1 == -1 for the 1st), 
 * so that start > end, the partition contains no elemens
 */
template<class ElemType, class ContainerType>
int* ArraySelfPartition<ElemType, ContainerType>::operator() (
	ContainerType &arr, 
	int arr_start,
	int arr_end,
	ElemPartOf<ElemType> **part_of, 
	int partition_count,
	NotifySwap* notify_swap
	) {

	if (partition_count > indexCapacity) {
		delete[] index;
		index = new int[partition_count];
		indexCapacity = partition_count;
	}
	
	int i, which_partition, j;
	ElemType elem;

	for (i = 0; i < partition_count; i ++) {
		index[i] = arr_start - 1;
	}

	for (i = arr_start; i <= arr_end; i ++) {

		// find the element belongs to which partition
		for (which_partition = 0; which_partition < partition_count; which_partition ++) 
			if ((*part_of[which_partition])(arr[i])) 
				break;

		// the new elem
		//elem = arr[i];

		// swap the elements
		for (j = partition_count - 1; j > which_partition; j --) {

			if (index[j] + 1 != index[j - 1] + 1) {

				elem = arr[index[j] + 1];
				arr[index[j] + 1] = arr[index[j - 1] + 1];
				arr[index[j - 1] + 1] = elem;

				if (notify_swap)
					(*notify_swap)(index[j] + 1, index[j - 1] + 1);
			}
			
			index[j] ++;
		}

		// assign the new elem to the last of the enlarged (by 1) partition
		index[which_partition] ++;
		//arr[index[which_partition]] = elem;
	}

	return index;
}

/* ----------------------------------------- */

/* quick sort in ascending order */
template<class T>
inline void hswap(T &a, T &b) {
	T temp;

	temp = a;
	a = b;
	b = temp;
}

template<class ElemType, class ContainerType> 
inline int partition2(ContainerType arr, int start, int end) {

	if (start >= end) 
		return start;

	// get the randomized pivot position
	int i = urand() % (end - start + 1);
	i += start;

	// swap the first element with the pivot
	hswap(arr[start], arr[i]);

	i = start;
	int j;

	for (j = start + 1; j <= end; j ++) {
		
		if (arr[j] < arr[start]) {
			hswap(arr[j], arr[i + 1]);
			i ++;
		}
	}

	hswap(arr[start], arr[i]);

	return i;
}

template<class ElemType, class ContainerType>
void recursive_partition(ContainerType arr, int start, int end) {

	if (start < 0)
		return;
	if (start >= end)
		return;

	int i;

	i = partition2<ElemType, ContainerType>(arr, start, end);
	recursive_partition<ElemType, ContainerType>(arr, start, i - 1);
	recursive_partition<ElemType, ContainerType>(arr, i + 1, end);
}

template<class ElemType, class ContainerType>
void h_quick_sort(ContainerType arr, int arr_count) {

	srand(0);
	recursive_partition<ElemType, ContainerType>(arr, 0, arr_count - 1);
}

/* merge two ascending array */
template<class ElemType, class ContainerType>
void merge_sorted_arr(
	ContainerType arr1, int arr1_count, 
	ContainerType arr2, int arr2_count, 
	ContainerType dst) {
	
	int i, j, k;

	for (i = 0, j = 0, k = 0; i < arr1_count || j < arr2_count; k ++) {
		if (j >= arr2_count || i < arr1_count && arr1[i] <= arr2[j]) {
			dst[k] = arr1[i];
			i ++;
		}
		else {
			dst[k] = arr2[j];
			j ++;
		}
	}
};

#endif //__H_ALGORITHM__