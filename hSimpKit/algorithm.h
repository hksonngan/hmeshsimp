/*
 *  algorithm
 *
 *  ht
 *  waytofall916@gmail.com
 */

#ifndef __H_ALGORITHM__
#define __H_ALGORITHM__


/* if the element is part of a concerning patch */
template<class ElemType>
class ElemPartOf {
public:
	virtual bool operator() (ElemType e) = 0;
};

/* ----------------------------------------- */

/* array self partition like what's in quicksort 
 * able to partition into arbitrary patches */
template<class ElemType, class ContainerType>
class ArraySelfPartition {
public:
	ArraySelfPartition(int _cap = 2);
	~ArraySelfPartition();
	int* operator() (ContainerType &arr, int arr_start, int arr_end, ElemPartOf<ElemType> **part_of, int partion_count);

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
	int partition_count) {

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
		elem = arr[i];

		// assign the fist elem before the shifting to the last elem after the shifting
		for (j = partition_count - 1; j > which_partition; j --) {
			arr[index[j] + 1] = arr[index[j - 1] + 1];
			index[j] ++;
		}

		// assign the new elem to the last of the enlarged (by 1) partition
		index[which_partition] ++;
		arr[index[which_partition]] = elem;
	}

	return index;
}

/* ----------------------------------------- */

#endif //__H_ALGORITHM__