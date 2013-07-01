#include "vol_set.h"
#include <iostream>
#include <fstream>
#include <stdio.h>

using std::ifstream;
using std::cout;
using std::endl;

int main() {
	ifstream fin("input.txt");
	if (!fin.good()) 
		return 0;

	char namebuf[1000];
	fin.getline(namebuf, 1000);

	VolumeSet vol_set;
	if(!vol_set.parseDataFile(namebuf, true, false))
		return 0;

	int x, y, z;
	while (true) {
		fin >> x >> y >> z;
		cout << "[" << x << ", " << y << ", " << z << "]: " << vol_set.getDense(x, y, z) << endl;
		if (fin.eof())
			break;
	}

	cout << endl << "press key ...";
	getchar();

	return 0;
}