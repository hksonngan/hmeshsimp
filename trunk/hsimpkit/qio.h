/**
 * COPYRIGHT NOTICE
 * Copyright (c) 2012, Institute of CG & CAD, Tsinghua University.
 * All Rights Reserved.
 * 
 * @file    QIO.h
 * @brief   QIO class definition.
 * 
 * This file defines the commonly used IO methods.
 * These methods includes reading/writing a text file or a binary file.
 * 
 * @version 1.0
 * @author  Jackie Pang, Tao Hou (Modify)
 * @e-mail: 15pengyi@gmail.com
 */

#ifndef QIO_H
#define QIO_H

#include <string>
//#include "../infrastructures/QStructure.h"

enum QDataFormat {
	DATA_UNKNOWN = 0,
	DATA_CHAR,
	DATA_UCHAR,	
	DATA_SHORT,
	DATA_USHORT,
	DATA_FLOAT,
	DATA_DOUBLE
};

enum QEndianness {
	ENDIAN_BIG      = 0,
	ENDIAN_LITTLE   = 1
};

class QIO
{
public:
    QIO();
    ~QIO();
	// layer start from 1
	// x is the voxel count of first dimension
	// y is the voxel count of second dimension
	// -ht
	static int getOffset(int layer, int x, int y, QDataFormat format) {
		int offset = 0;
		switch (format) {
			case DATA_CHAR:
			case DATA_UCHAR:
				offset = 1;
				break;
			case DATA_SHORT:
			case DATA_USHORT:
				offset = 2;
				break;
			case DATA_FLOAT:
				offset = 4;
				break;
			case DATA_DOUBLE:
				offset = 8;
				break;
		}

		offset *= x * y * (layer - 1);
		return offset;
	}
    static bool getFileContent(std::string fileName, std::string &content);
    static bool getFileData(std::string fileName, void *data, unsigned int size);
	static bool getFileDataOff(std::string fileName, void *data, unsigned int size, int offset);
    static bool saveFileData(std::string fileName, void *data, unsigned int size);
private:

};

#endif  // QIO_H