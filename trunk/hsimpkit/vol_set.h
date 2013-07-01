/*
 *	Read Volume Set File
 *
 *  Author: Ht
 *  Email : waytofall916 at gmail dot com
 *
 *  Copyright (C) Ht. All rights reserved.
 */

#ifndef __VOLUME_SET_H__
#define __VOLUME_SET_H__

#include <string>
#include <fstream>
#include <iostream>
#include <vector>
#include "qio.h"
#include "mc.h"
#include "data_type.h"
#include "trivial.h"

using namespace MC;
using std::vector;

typedef struct _UINT4 {
	unsigned int s[4];
} UINT4;

typedef struct _FLOAT4 {
	float s[4];
} FLOAT4;

typedef char Byte;

// Read Volume Set File
// Can read layer by layer (without loading the whole volume data file)
// or perform normal reading (with loading the whole volume data file)
class VolumeSet {
public:
	// Data file
	std::string dataFilePath, dataFileName, objectFileName;

	// Volumetric Data
	EndianOrder fileEndian;
	EndianOrder systemEndian;
	QDataFormat format;
	unsigned int formatSize;
    UINT4 sampleStride;
	UINT4 volumeSize;
	FLOAT4 thickness;

	std::ifstream *pfin;
	UINT4 cursor; // cube index
	Byte *upper, *lower;
	// use vector to allocate memory
	//vector<Byte> *upperVec, *lowerVec;
	Byte *_data;
	//vector<Byte> *dataVec;
    int layerStride;
	bool layeredRead;
    bool DATA_ARR_ALLOC_IN_THIS_OBJECT;

public:
	VolumeSet();
	~VolumeSet();
	bool parseDataFile(const std::string &name, bool allocMem = true, bool _layeredRead = true);
	bool nextCube(GRIDCELL &cube);
	bool hasNext();
	void clear();
	Byte* getData() { return _data; }
	double getDense(unsigned int i, unsigned int j, unsigned int k);
	inline int memSize();
    VolumeSet& operator=(const VolumeSet& volSet);
	void memCheck();

    inline bool rightMost(const UINT4 &cubeIndex);
    inline bool backMost(const UINT4 &cubeIndex);
    inline bool downMost(const UINT4 &cubeIndex);
    inline bool rightBackMost(const UINT4 &cubeIndex);
    inline bool rightDownMost(const UINT4 &cubeIndex);
    inline bool backDownMost(const UINT4 &cubeIndex);
    inline bool rightBackDownMost(const UINT4 &cubeIndex);

private:
	void trim(std::string &s);
    bool readFirstLayer(Byte *layer);
	bool readNextLayer(Byte *layer);
	bool readData(Byte *d, unsigned int size);
	void getXYZ(XYZ &v, unsigned int i, unsigned int j, unsigned int k);
	double getDense(Byte* p, unsigned int i, unsigned int j);
	inline double getVoxelData(Byte *p);
};

inline double VolumeSet::getVoxelData(Byte *p) {
	switch(format) {
	case DATA_CHAR:
		return *((char*)p);
	case DATA_UCHAR:
		return *((unsigned char*)p);
	case DATA_SHORT:
		return *((short*)p);
	case DATA_USHORT:
		return *((unsigned short*)p);
	case DATA_FLOAT:
		return *((float*)p);
	}
    return 0;
}

inline int VolumeSet::memSize() { 
	std::cout << "#volume data size: " << volumeSize.s[0] * volumeSize.s[1] * volumeSize.s[2] * formatSize << std::endl;
	return volumeSize.s[0] * volumeSize.s[1] * volumeSize.s[2] * formatSize; 
}

bool VolumeSet::rightMost(const UINT4 &cubeIndex) {
    return cubeIndex.s[0] + sampleStride.s[0] >= volumeSize.s[0] - 1;
}

bool VolumeSet::backMost(const UINT4 &cubeIndex) {
    return cubeIndex.s[1] + sampleStride.s[1] >= volumeSize.s[1] - 1;
}

bool VolumeSet::downMost(const UINT4 &cubeIndex) {
    return cubeIndex.s[2] + sampleStride.s[2] >= volumeSize.s[2] - 1;
}

bool VolumeSet::rightBackMost(const UINT4 &cubeIndex) {
    return rightMost(cubeIndex) && backMost(cubeIndex);
}

bool VolumeSet::rightDownMost(const UINT4 &cubeIndex) {
    return rightMost(cubeIndex) && downMost(cubeIndex);
}

bool VolumeSet::backDownMost(const UINT4 &cubeIndex) {
    return backMost(cubeIndex) && downMost(cubeIndex);
}

bool VolumeSet::rightBackDownMost(const UINT4 &cubeIndex) {
    return rightMost(cubeIndex) && backMost(cubeIndex) && downMost(cubeIndex);
}

#endif