#include <stdlib.h>
#include <time.h>
#include "algorithm.h"
#include <iostream>

using std::cout;
using std::endl;

class Belong1 : public ElemPartOf<int>
{
public:
	virtual bool operator() (int i) {
		return i >= 0 && i < 10;
	}
};

class Belong2 : public ElemPartOf<int>
{
public:
	virtual bool operator() (int i) {
		return i >= 10 && i < 20;
	}
};

class Belong3 : public ElemPartOf<int>
{
public:
	virtual bool operator() (int i) {
		return i >= 20 && i < 30;
	}
};

class Belong4 : public ElemPartOf<int>
{
public:
	virtual bool operator() (int i) {
		return i >= 30 && i < 40;
	}
};

class Belong5 : public ElemPartOf<int>
{
public:
	virtual bool operator() (int i) {
		return i >= 40 && i < 50;
	}
};

class Belong6 : public ElemPartOf<int>
{
public:
	virtual bool operator() (int i) {
		return i >= 50 && i < 60;
	}
};

class Belong7 : public ElemPartOf<int>
{
public:
	virtual bool operator() (int i) {
		return i >= 60 && i < 70;
	}
};

class Belong8 : public ElemPartOf<int>
{
public:
	virtual bool operator() (int i) {
		return i >= 70 && i < 80;
	}
};

int main(int argc, char** argv)
{
	int i, j, k, l, n[50], *index;

	cout << "array: " << endl;
	srand ( time(NULL) );
	for (i = 0; i < 50; i ++) {
		n[i] = rand() % 80;
		cout << n[i] << " ";
	}
	cout << endl << endl;

	Belong1 b1;
	Belong2 b2;
	Belong3 b3;
	Belong4 b4;
	Belong5 b5;
	Belong6 b6;
	Belong7 b7;
	Belong8 b8;

	ElemPartOf<int>* partof[] = {&b1, &b2, &b3, &b4, &b5, &b6, &b7, &b8};

	ArraySelfPartition<int, int[50]> partition;
	index = partition(n, 10, 49, partof, 8);

	for (i = 0; i < 8; i ++) {
		if (i == 0) 
			j = 10;
		else
			j = index[i - 1] + 1;

		k = index[i];

		cout << "partition " << i << " start at " << j << " end at " << k << ", elements: " << endl;
		for (l = j; l <= k; l ++) {
			cout << n[l] << " ";
		}
		cout << endl;
	}
}