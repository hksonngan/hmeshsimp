#include "random.h"
#include <iostream>
#include <fstream>
#include "h_algorithm.h"
#include "h_time.h"

using namespace std;

#define SIZE 0
#define SIZE2 10

int main() {

	//unsigned *arr = new unsigned[SIZE];
	//unsigned *arr2 = new unsigned[SIZE2];
	unsigned *dst = new unsigned[SIZE + SIZE2];
	ofstream fout("sort.txt");

	unsigned arr[] = {15,1,2,3,4,5,6,7,8,9,10};
	unsigned arr2[] = {11,12,13,14,15,16,17,18,19,20};

	usrand(0);
	//for (int i = 0; i < SIZE; i ++) {
	//	arr[i] = urand();
	//	//fout << arr[i] << endl;
	//}
	//for (int i = 0; i < SIZE2; i ++) {
	//	arr2[i] = urand();
	//}

	HTime htime;
	h_quick_sort<unsigned, unsigned*>(arr, SIZE);
	h_quick_sort<unsigned, unsigned*>(arr2, SIZE2);
	merge_sorted_arr<unsigned, unsigned*>(arr, SIZE, arr2, SIZE2, dst);
	//fout << htime.printElapseSec() << endl << endl;

	for (int i = 0; i < SIZE; i ++) {
		fout << arr[i] << endl;
	}
	fout << endl << "===================" << endl;


	for (int i = 0; i < SIZE2; i ++) {
		fout << arr2[i] << endl;
	}
	fout << endl << "===================" << endl;

	for (int i = 0; i < SIZE + SIZE2; i ++) {
		fout << dst[i] << endl;
	}
	fout << endl << "===================" << endl;

	delete[] arr;
	delete[] arr2;
	delete[] dst;
}