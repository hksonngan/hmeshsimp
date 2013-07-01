#include "vol_set.h"
#include <sstream>
#include <iostream>
#include "trivial.h"

VolumeSet::VolumeSet() {
	upper = NULL;
	lower = NULL;
	_data = NULL;
	//upperVec = NULL;
	//lowerVec = NULL;
	//dataVec = NULL;
    DATA_ARR_ALLOC_IN_THIS_OBJECT = false;
    sampleStride.s[0] = 1;
    sampleStride.s[1] = 1;
    sampleStride.s[2] = 1;
	systemEndian = getSystemEndianMode();
    pfin = NULL;
	thickness.s[0] = 1.0f;
	thickness.s[1] = 1.0f;
	thickness.s[2] = 1.0f;
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
bool VolumeSet::parseDataFile(const std::string &name, bool allocMem, bool _layeredRead) {
	clear();
	fileEndian = systemEndian;

	dataFileName = name;
	int position = dataFileName.find_last_of("\\");
	if (position == std::string::npos) position = dataFileName.find_last_of("/");
	if (position == std::string::npos) 
		dataFilePath = "";
	else
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
			if ((position = line.find(':')) == std::string::npos) {
				error = true;
				break;
			}
			objectFileName = line.substr(position + 1);
			trim(objectFileName);
		} else if ((position = line.find("Resolution")) != std::string::npos) {
			if ((position = line.find(':')) == std::string::npos) {
				error = true;
				break;
			}
			buffer << line.substr(position + 1);
			unsigned int x = 0, y = 0, z = 0;
			buffer >> x >> y >> z;
			if (x <= 0 || y <= 0 || z <= 0) {
				error = true;
				break;
			}
			volumeSize.s[0] = x;
			volumeSize.s[1] = y;
			volumeSize.s[2] = z;
		} else if ((position = line.find("SliceThickness")) != std::string::npos) {
			if ((position = line.find(':')) == std::string::npos) {
				error = true;
				break;
			}
			buffer << line.substr(position + 1);
			float x = 0.0, y = 0.0, z = 0.0;
			buffer >> x >> y >> z;
			if (x <= 0.0 || y <= 0.0 || z <= 0.0) {
				error = true;
				break;
			}
			thickness.s[0] = x;
			thickness.s[1] = y;
			thickness.s[2] = z;
		} else if ((position = line.find("Format")) != std::string::npos) {
			if ((position = line.find(':')) == std::string::npos) {
				error = true;
				break;
			}

            string valstr = getNextWordBetweenSpace(position + 1, line);
			if (valstr.compare("CHAR") == 0) {
				format = DATA_CHAR;
				formatSize = sizeof(char);
			} else if (valstr.compare("UCHAR") == 0) {
				format = DATA_UCHAR;
				formatSize = sizeof(unsigned char);
			} else if (valstr.compare("SHORT") == 0) {
				format = DATA_SHORT;
				formatSize = sizeof(short);
			} else if (valstr.compare("USHORT") == 0) {
				format = DATA_USHORT;
				formatSize = sizeof(unsigned short);
			} else if (valstr.compare("FLOAT") == 0) {
				format = DATA_FLOAT;
				formatSize = sizeof(float);
			} else {
				std::cerr << " > ERROR: cannot process data other than of CHAR, UCHA USHORT format." << std::endl;
				error = true;
			}
		} else if ((position = line.find("Endian")) != std::string::npos) {
			if ((position = line.find(':')) == std::string::npos) {
				error = true;
				break;
			}

			if ((position = line.find("BIG-ENDIAN")) != std::string::npos) {
				fileEndian = H_BIG_ENDIAN;
			} else if ((position = line.find("LITTLE-ENDIAN")) != std::string::npos) {
				fileEndian = H_LITTLE_ENDIAN;
			} else {
				std::cerr << " > ERROR: cannot process endian other than of BIG-ENDIAN and LITTLE-ENDIAN format. Set to system endian" << std::endl;
				error = true;
			}
		} else {
			std::cerr << " > WARNING: skipping line \"" << line << "\"." << std::endl;
		}
	}

	if (error) {
		std::cerr << " > ERROR: parsing \"" << line << "\"." << std::endl;
		return false;
	}

    if (pfin)
        delete pfin;
    pfin = new std::ifstream();
	pfin->open((dataFilePath + objectFileName).c_str(), std::ios::binary);
	if (!pfin->good()) {
		std::cerr << " > ERROR: cannot open .raw file" << std::endl;
		return false;
	}

	cursor.s[0] = cursor.s[1] = cursor.s[2] = 0;
    this->layeredRead = _layeredRead;

	if (allocMem) {
		if (layeredRead) {
			upper = new Byte[volumeSize.s[0] * volumeSize.s[1] * formatSize];
			lower = new Byte[volumeSize.s[0] * volumeSize.s[1] * formatSize];
			//upperVec = new vector<Byte>();
			//upperVec->resize(volumeSize.s[0] * volumeSize.s[1] * formatSize);
			//upper = upperVec->data();
			//lowerVec = new vector<Byte>();
			//lowerVec->resize(volumeSize.s[0] * volumeSize.s[1] * formatSize);
			//lower = lowerVec->data();
		} else {
			_data = new Byte[volumeSize.s[0] * volumeSize.s[1] * volumeSize.s[2] * formatSize];
			//dataVec = new vector<Byte>();
			//dataVec->resize(volumeSize.s[0] * volumeSize.s[1] * volumeSize.s[2] * formatSize);
			//_data = dataVec->data();
			readData(_data, volumeSize.s[0] * volumeSize.s[1] * volumeSize.s[2] * formatSize);
		}
        DATA_ARR_ALLOC_IN_THIS_OBJECT = true;
	}

	std::cerr << std::endl;

	return true;
}

void VolumeSet::getXYZ(XYZ &v, unsigned int i, unsigned int j, unsigned int k) {
	v.x = i * thickness.s[0];
	v.y = j * thickness.s[1];
	v.z = k * thickness.s[2];
}

double VolumeSet::getDense(Byte* p, unsigned int i, unsigned int j) {
	p += (i + j * volumeSize.s[0]) * formatSize;
	return getVoxelData(p);	
}

double VolumeSet::getDense(unsigned int i, unsigned int j, unsigned int k) {
	int elem_index = (i + j * volumeSize.s[0] + k * volumeSize.s[0] * volumeSize.s[1]);
	int offset = elem_index * formatSize;
	Byte* p = _data + (i + j * volumeSize.s[0] + k * volumeSize.s[0] * volumeSize.s[1]) * formatSize;
	return getVoxelData(p);
}

bool VolumeSet::nextCube(GRIDCELL &cube) {
    if (layeredRead) {
	    if (cursor.s[0] == 0 && cursor.s[1] == 0) {
		    if (cursor.s[2] == 0) {
			    if (!readFirstLayer(upper))
				    return false;
                layerStride = sampleStride.s[2];
                if (layerStride > volumeSize.s[2] - 1)
                    layerStride = volumeSize.s[2] - 1;
			    if (!readNextLayer(lower))
				    return false;
		    } else {
			    std::swap(upper, lower);
			    if (!readNextLayer(lower))
				    return false;
		    }
	    }
    }

	// in the figure of http://paulbourke.net/geometry/polygonise/
	// the x axis is (0, 1), y axis is (0, 3), z axis is (0, 4)
	// the traversal order of the cues is: 
	//   0 -> 1 in x dimension
	//   0 -> 3 in y dimension
	//   4 -> 0 in z dimension
	// so 4 is the start vertex
    unsigned int x2 = cursor.s[0] + sampleStride.s[0];
    if (x2 > volumeSize.s[0] - 2)
        x2 = volumeSize.s[0] - 2;
    unsigned int y2 = cursor.s[1] + sampleStride.s[1];
    if (y2 > volumeSize.s[1] - 2)
        y2 = volumeSize.s[1] - 2;
    unsigned int z2 = cursor.s[2] + sampleStride.s[2];
    if (z2 > volumeSize.s[2] - 2)
        z2 = volumeSize.s[2] - 2;

	/*  vert of cube  x of voxel   y of voxel   z of voxel  */
	getXYZ(cube.p[0], cursor.s[0], cursor.s[1], z2);
	getXYZ(cube.p[1], x2,          cursor.s[1], z2);
	getXYZ(cube.p[2], x2,          y2,          z2);
	getXYZ(cube.p[3], cursor.s[0], y2,          z2);
	getXYZ(cube.p[4], cursor.s[0], cursor.s[1], cursor.s[2]);
	getXYZ(cube.p[5], x2,          cursor.s[1], cursor.s[2]);
	getXYZ(cube.p[6], x2,          y2,          cursor.s[2]);
	getXYZ(cube.p[7], cursor.s[0], y2,          cursor.s[2]);

    if (layeredRead) {
	    cube.val[0] = getDense(lower, cursor.s[0], cursor.s[1]);
	    cube.val[1] = getDense(lower, x2,          cursor.s[1]);
	    cube.val[2] = getDense(lower, x2,          y2);
	    cube.val[3] = getDense(lower, cursor.s[0], y2);
	    cube.val[4] = getDense(upper, cursor.s[0], cursor.s[1]);
	    cube.val[5] = getDense(upper, x2,          cursor.s[1]);
	    cube.val[6] = getDense(upper, x2,          y2);
        cube.val[7] = getDense(upper, cursor.s[0], y2); 
    } else {
        cube.val[0] = getDense(cursor.s[0], cursor.s[1], z2);
        cube.val[1] = getDense(x2,          cursor.s[1], z2);
        cube.val[2] = getDense(x2,          y2,          z2);
        cube.val[3] = getDense(cursor.s[0], y2,          z2);
        cube.val[4] = getDense(cursor.s[0], cursor.s[1], cursor.s[2]);
        cube.val[5] = getDense(x2,          cursor.s[1], cursor.s[2]);
        cube.val[6] = getDense(x2,          y2,          cursor.s[2]);
        cube.val[7] = getDense(cursor.s[0], y2,          cursor.s[2]); 
    }

	cursor.s[0] += sampleStride.s[0];
	// max cube index is one less than max slice index
	if (cursor.s[0] >= volumeSize.s[0] - 2) {
		cursor.s[0] = 0;
		cursor.s[1] += sampleStride.s[1];
		if (cursor.s[1] >= volumeSize.s[1] - 2) {
			cursor.s[1] = 0;
			cursor.s[2] += sampleStride.s[2];
            if (cursor.s[2] + sampleStride.s[2] > volumeSize.s[2] - 1)
                layerStride = volumeSize.s[2] - 1 - cursor.s[2];
		}
	}

	return true;
}

bool VolumeSet::hasNext() {
	return cursor.s[2] < volumeSize.s[2] - 1;
}

void VolumeSet::clear() {
    cursor.s[0] = cursor.s[1] = cursor.s[2] = 0;
	if (pfin) {
        delete pfin;
		pfin = NULL;
	}
    if (DATA_ARR_ALLOC_IN_THIS_OBJECT == false)
        return;
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
	//if (upperVec) 
	//	delete upperVec;
	//if (lowerVec)
	//	delete lowerVec;
	//if (dataVec)
	//	delete dataVec;
}

bool VolumeSet::readFirstLayer(Byte *layer) {
    int size = volumeSize.s[0] * volumeSize.s[1] * formatSize;
    pfin->read(layer, size);
    int readSize = pfin->gcount();

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

bool VolumeSet::readNextLayer(Byte *layer) {
    int size = volumeSize.s[0] * volumeSize.s[1] * formatSize;
    pfin->seekg(size * (layerStride - 1), std::ios_base::cur);
	pfin->read(layer, size);
	int readSize = pfin->gcount();

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
	pfin->read(d, size);
	int readSize = pfin->gcount();

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

VolumeSet& VolumeSet::operator=(const VolumeSet& volSet) {
    memcpy(this, const_cast<VolumeSet *>(&volSet), sizeof(VolumeSet));
    DATA_ARR_ALLOC_IN_THIS_OBJECT = false;
    pfin = NULL;
	//upperVec = NULL;
	//lowerVec = NULL;
	//dataVec = NULL;
    return *this;
}

void VolumeSet::memCheck() {
	for (int k = 0; k < volumeSize.s[2]; k ++)
		for (int j = 0; j < volumeSize.s[1]; j ++)
			for (int i = 0; i < volumeSize.s[0]; i ++)
				double val = getVoxelData(_data +  (i + j * volumeSize.s[0] + k * volumeSize.s[0] * volumeSize.s[1]) * formatSize);
}