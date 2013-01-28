/*
 *	Read Volume Set File
 *	Modified from Jackie Pang in QRendering
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
#include "qio.h"
#include "mc.h"
#include "data_type.h"
#include "trivial.h"

using namespace MC;

typedef struct {
	unsigned int s[4];
} UINT4;

typedef struct {
	float s[4];
} FLOAT4;

typedef char Byte;

class VolumeSet {
public:
	// Data file
	std::string dataFilePath, dataFileName, objectFileName;

	// Volumetric Data
	EndianOrder fileEndian;
	EndianOrder systemEndian;
	QDataFormat format;
	unsigned int formatSize;
	UINT4 volumeSize;
	FLOAT4 thickness;

	std::ifstream fin;
	UINT4 cursor; // cube index
	Byte *upper, *lower;
	Byte *_data;

public:
	VolumeSet();
	~VolumeSet();
	bool parseDataFile(const std::string &name, bool allocMem = true);
	bool nextCube(GRIDCELL &cube);
	bool hasNext();
	void clear();

private:
	void trim(std::string &s);
	bool readNextLayer(Byte *layer);
	bool readData(Byte *d, unsigned int size);
	void getXYZ(XYZ &v, unsigned int i, unsigned int j, unsigned int k);
	double getDense(Byte* p, unsigned int i, unsigned int j);
	double getDense2(unsigned int i, unsigned int j, unsigned int k);
};

#endif