#include "h_dynamarray.h"
#include <iostream>
#include <fstream>

using namespace std;

int main() {

	HDynamArray<unsigned> arr1, arr2;

	ofstream fout("merge.txt");

	arr1.randGen(10, 10);
	arr2.randGen(10, 10);

	fout << arr1 << endl
		<< "===========" << endl
		<< arr2 << endl
		<< "===========" << endl << endl;

	arr1.merge(arr2);

	fout << arr1;
}