#include "p_adap_iso.h"
#include <string>
#include <iostream>
#include <fstream>
#include <stdlib.h>

#define __forceinline__ 
#define __device__ 
#include "_child_config.h"

using std::string;
using std::cout;
using std::cerr;
using std::endl;

void makeSimpleRaw() {
	unsigned char data[] = {
		0,1,0, 1,0,1, 0,1,0, 
		1,0,1, 0,1,0, 1,0,1,
		0,1,0, 1,0,1, 0,1,0
	};

	std::ofstream fout ("simple.raw", std::ios::binary | std::ios::out);
	fout.write(reinterpret_cast<char*>(data), sizeof(data));
};

int main(int argc, char** argv)
{
	string info;

	//makeSimpleRaw();
	//return 0;

	extern int testCudaPrintf();
	//testCudaPrintf();

	char fnamebuf[2000];
	char buf[100];
	int start_depth;
	float error_thresh;
	float isovalue;

	std::ifstream fin("params.txt");

	fin.getline(fnamebuf, 2000);
	fin.getline(buf, 100);
	start_depth = atoi(buf);
	fin.getline(buf, 100);
	error_thresh = atof(buf);
	fin.getline(buf, 100);
	isovalue = atof(buf);

	if (!pAdaptiveIso(fnamebuf, start_depth, error_thresh, isovalue, info)) {
		cerr << "error occurred: " << info << endl;
		exit(-1);
	}

	std::ofstream fout("padaplog.txt",  std::fstream::out | std::fstream::app);

	fout << endl 
		<< "volume file name: " << fnamebuf << endl
		<< "start depth: " << start_depth << endl
		<< "error thresh: " << error_thresh << endl
		<< "iso value: " << isovalue << endl
		<< info << endl;

	cout << endl 
		<< "volume file name: " << fnamebuf << endl
		<< "start depth: " << start_depth << endl
		<< "error thresh: " << error_thresh << endl
		<< "iso value: " << isovalue << endl
		<< info << endl;

	return 0;
}

//unsigned char config = 0, x_conf, y_conf, z_conf;

//childConfigSetXLeft(config);
//childConfigSetYRight(config);
//childConfigSetZWhole(config);

//getXConfig(config, x_conf);
//getYConfig(config, y_conf);
//getZConfig(config, z_conf);

//childConfigSetXRight(config);
//childConfigSetYLeft(config);
//childConfigSetZRight(config);

//getXConfig(config, x_conf);
//getYConfig(config, y_conf);
//getZConfig(config, z_conf);

//childConfigSetXWhole(config);
//childConfigSetYWhole(config);
//childConfigSetZLeft(config);

//getXConfig(config, x_conf);
//getYConfig(config, y_conf);
//getZConfig(config, z_conf);