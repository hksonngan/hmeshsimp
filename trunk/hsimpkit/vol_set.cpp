#include "vol_set.h"
#include <sstream>
#include <iostream>

VolumeSet::VolumeSet() {
	upper = NULL;
	lower = NULL;
	_data = NULL;
	systemEndian = getSystemEndianMode();
}

VolumeSet::~VolumeSet() {
	clear();
}

void VolumeSet::trim(std::string &s)
{
	if (!s.empty())
	{
		int found = s.find_first_of('\t');
		while (found != std::string::npos)
		{
			s.replace(found, 1, " ");
			found = s.find_first_of('\t', found + 1);
		}
		s.erase(0, s.find_first_not_of(' '));
		s.erase(s.find_last_not_of(' ') + 1);
	}
}

// parse .dat file
bool VolumeSet::parseDataFile(const std::string &name, bool allocMem) {
	clear();
	fileEndian = systemEndian;

	dataFileName = name;
	int position = dataFileName.find_last_of("\\");
	if (position == std::string::npos) position = dataFileName.find_last_of("/");
	if (position == std::string::npos) position = dataFileName.size() - 1;
	dataFilePath = dataFileName.substr(0, position + 1);

	std::string dataFileContent, line;
	if (!QIO::getFileContent(name, dataFileContent)) return false;

	std::stringstream data(dataFileContent, std::stringstream::in);
	bool error = false;
	position = std::string::npos;
	while (!data.eof()) {
		getline(data, line);
		std::stringstream buffer(std::stringstream::in | std::stringstream::out);
		if ((position = line.find("ObjectFileName")) != std::string::npos) {
			if ((position = line.find(':')) == std::string::npos)
			{
				error = true;
				break;
			}
			objectFileName = line.substr(position + 1);
			trim(objectFileName);
		}
		else if ((position = line.find("Resolution")) != std::string::npos) {
			if ((position = line.find(':')) == std::string::npos)
			{
				error = true;
				break;
			}
			buffer << line.substr(position + 1);
			unsigned int x = 0, y = 0, z = 0;
			buffer >> x >> y >> z;
			if (x <= 0 || y <= 0 || z <= 0)
			{
				error = true;
				break;
			}
			volumeSize.s[0] = x;
			volumeSize.s[1] = y;
			volumeSize.s[2] = z;
		}
		else if ((position = line.find("SliceThickness")) != std::string::npos) {
			if ((position = line.find(':')) == std::string::npos)
			{
				error = true;
				break;
			}
			buffer << line.substr(position + 1);
			float x = 0.0, y = 0.0, z = 0.0;
			buffer >> x >> y >> z;
			if (x <= 0.0 || y <= 0.0 || z <= 0.0)
			{
				error = true;
				break;
			}
			thickness.s[0] = x;
			thickness.s[1] = y;
			thickness.s[2] = z;
		}
		else if ((position = line.find("Format")) != std::string::npos) {
			if ((position = line.find(':')) == std::string::npos)
			{
				error = true;
				break;
			}
			if ((position = line.find("UCHAR")) != std::string::npos)
			{
				format = DATA_UCHAR;
				formatSize = 1;
			}
			else if ((position = line.find("USHORT")) != std::string::npos)
			{
				format = DATA_USHORT;
				formatSize = 2;
			}
			else if ((position = line.find("FLOAT")) != std::string::npos)
			{
				format = DATA_FLOAT;
				formatSize = 4;
			}
			else
			{
				std::cerr << " > ERROR: cannot process data other than of UCHAR and USHORT format." << std::endl;
				error = true;
			}
		}
		//else if ((position = line.find("Window")) != std::string::npos)
		//{
		//	if ((position = line.find(':')) == std::string::npos)
		//	{
		//		error = true;
		//		break;
		//	}
		//	buffer << line.substr(position + 1);
		//	buffer >> windowWidth >> windowLevel;
		//	if (windowWidth <= 0.0f)
		//	{
		//		error = true;
		//		break;
		//	}
		//}
		else if ((position = line.find("Endian")) != std::string::npos) {
			if ((position = line.find(':')) == std::string::npos)
			{
				error = true;
				break;
			}
			if ((position = line.find("BIG-ENDIAN")) != std::string::npos)
			{
				fileEndian = H_BIG_ENDIAN;
			}
			else if ((position = line.find("LITTLE-ENDIAN")) != std::string::npos)
			{
				fileEndian = H_LITTLE_ENDIAN;
			}
			else
			{
				std::cerr << " > ERROR: cannot process endian other than of BIG-ENDIAN and LITTLE-ENDIAN format. Set to system endian" << std::endl;
				error = true;
			}
		}
		else {
			std::cerr << " > WARNING: skipping line \"" << line << "\"." << std::endl;
		}
	}

	if (error) {
		std::cerr << " > ERROR: parsing \"" << line << "\"." << std::endl;
		return false;
	}

	fin.open((dataFilePath + objectFileName).c_str(), std::ios::binary);
	if (!fin.good()) {
		std::cerr << " > ERROR: cannot open .raw file" << std::endl;
		return false;
	}

	cursor.s[0] = cursor.s[1] = cursor.s[2] = 0;

	if (allocMem) {
		upper = new Byte[volumeSize.s[0] * volumeSize.s[1] * formatSize];
		lower = new Byte[volumeSize.s[0] * volumeSize.s[1] * formatSize];
		//_data = new Byte[volumeSize.s[0] * volumeSize.s[1] * volumeSize.s[2] * formatSize];
		//readData(_data, volumeSize.s[0] * volumeSize.s[1] * volumeSize.s[2] * formatSize);
	}

	std::cerr << std::endl;

	return true;
}

void VolumeSet::getXYZ(XYZ &v, unsigned int i, unsigned int j, unsigned int k) {
	v.x = i * thickness.s[0];
	v.y = j * thickness.s[1];
	v.z = -(k * thickness.s[2]);
}

double VolumeSet::getDense(Byte* p, unsigned int i, unsigned int j) {
	p += (i + j * volumeSize.s[0]) * formatSize;

	switch(format) {
	case DATA_UCHAR:
		return *((unsigned char*)p);
	case DATA_USHORT:
		return *((unsigned short*)p);
	case DATA_FLOAT:
		return *((float*)p);
	}
}

double VolumeSet::getDense2(unsigned int i, unsigned int j, unsigned int k) {
	Byte* p = _data + (i + j * volumeSize.s[0] + k * volumeSize.s[0] * volumeSize.s[1]) * formatSize;

	switch(format) {
	case DATA_UCHAR:
		return *((unsigned char*)p);
	case DATA_USHORT:
		return *((unsigned short*)p);
	case DATA_FLOAT:
		return *((float*)p);
	}
}

bool VolumeSet::nextCube(GRIDCELL &cube) {
	if (cursor.s[0] == 0 && cursor.s[1] == 0) {
		if (cursor.s[2] == 0) {
			if (!readNextLayer(upper))
				return false;
			if (!readNextLayer(lower))
				return false;
		} else {
			std::swap(upper, lower);
			if (!readNextLayer(lower))
				return false;
		}
	}

	// in the figure of http://paulbourke.net/geometry/polygonise/
	// the x axis is (0, 1), y axis is (0, 3), z axis is (0, 4)
	// the traversal order of the cues is: 
	//   0 -> 1 in x dimension
	//   0 -> 3 in y dimension
	//   4 -> 0 in z dimension
	// so 4 is the start vertex

	/*  vert of cube   x of voxel     y of voxel     z of voxel  */
	getXYZ(cube.p[0], cursor.s[0],   cursor.s[1],   cursor.s[2]+1);
	getXYZ(cube.p[1], cursor.s[0]+1, cursor.s[1],   cursor.s[2]+1);
	getXYZ(cube.p[2], cursor.s[0]+1, cursor.s[1]+1, cursor.s[2]+1);
	getXYZ(cube.p[3], cursor.s[0],   cursor.s[1]+1, cursor.s[2]+1);
	getXYZ(cube.p[4], cursor.s[0],   cursor.s[1],   cursor.s[2]);
	getXYZ(cube.p[5], cursor.s[0]+1, cursor.s[1],   cursor.s[2]);
	getXYZ(cube.p[6], cursor.s[0]+1, cursor.s[1]+1, cursor.s[2]);
	getXYZ(cube.p[7], cursor.s[0],   cursor.s[1]+1, cursor.s[2]);

	cube.val[0] = getDense(lower, cursor.s[0],   cursor.s[1]);
	cube.val[1] = getDense(lower, cursor.s[0]+1, cursor.s[1]);
	cube.val[2] = getDense(lower, cursor.s[0]+1, cursor.s[1]+1);
	cube.val[3] = getDense(lower, cursor.s[0],   cursor.s[1]+1);
	cube.val[4] = getDense(upper, cursor.s[0],   cursor.s[1]);
	cube.val[5] = getDense(upper, cursor.s[0]+1, cursor.s[1]);
	cube.val[6] = getDense(upper, cursor.s[0]+1, cursor.s[1]+1);
	cube.val[7] = getDense(upper, cursor.s[0],   cursor.s[1]+1);

	//cube.val[0] = getDense2(cursor.s[0],   cursor.s[1],   cursor.s[2]+1);
	//cube.val[1] = getDense2(cursor.s[0]+1, cursor.s[1],   cursor.s[2]+1);
	//cube.val[2] = getDense2(cursor.s[0]+1, cursor.s[1]+1, cursor.s[2]+1);
	//cube.val[3] = getDense2(cursor.s[0],   cursor.s[1]+1, cursor.s[2]+1);
	//cube.val[4] = getDense2(cursor.s[0],   cursor.s[1],   cursor.s[2]);
	//cube.val[5] = getDense2(cursor.s[0]+1, cursor.s[1],   cursor.s[2]);
	//cube.val[6] = getDense2(cursor.s[0]+1, cursor.s[1]+1, cursor.s[2]);
	//cube.val[7] = getDense2(cursor.s[0],   cursor.s[1]+1, cursor.s[2]);

	cursor.s[0] ++;
	// max cube index is one less than max slice index
	if (cursor.s[0] == volumeSize.s[0] - 1) {
		cursor.s[0] = 0;
		cursor.s[1] ++;
		if (cursor.s[1] == volumeSize.s[1] - 1) {
			cursor.s[1] = 0;
			cursor.s[2] ++;
		}
	}

	return true;
}

bool VolumeSet::hasNext() {
	return cursor.s[2] < volumeSize.s[2] - 1;
}

void VolumeSet::clear() {
	fin.close();
	fin.clear();
	if (upper) {
		delete upper;
		upper = NULL;
	}
	if (lower) {
		delete lower;
		lower = NULL;
	}
	if (_data) {
		delete _data;
		_data = NULL;
	}
}

bool VolumeSet::readNextLayer(Byte *layer) {
	int size = volumeSize.s[0] * volumeSize.s[1] * formatSize;
	fin.read(layer, size);
	int readSize = fin.gcount();

	if (readSize < size) {
		std::cerr << "#ERROR: read file error" << std::endl;
		return false;
	}

	if (fileEndian != systemEndian) {
		Byte *p = layer;
		int i = 0;
		for (; i < volumeSize.s[0] * volumeSize.s[1]; i ++, p += formatSize) 
			switchBytes(p, formatSize);
	}

	return true;
}

bool VolumeSet::readData(Byte *d, unsigned int size) {
	fin.read(d, size);
	int readSize = fin.gcount();

	if (readSize < size) {
		std::cerr << "#ERROR: read file error" << std::endl;
		return false;
	}

	if (fileEndian != systemEndian) {
		Byte *p = d;
		int i = 0;
		for (; i < volumeSize.s[0] * volumeSize.s[1]; i ++, p += formatSize) 
			switchBytes(p, formatSize);
	}

	return true;
}