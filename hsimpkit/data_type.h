/*
 *
 *  Author: Ht
 *  Email : waytofall916 at gmail dot com
 *
 *  Copyright (C) Ht. All rights reserved.
 */

#ifndef _H_DATA_TYPE_INC_
#define _H_DATA_TYPE_INC_

enum DATA_TYPE { 
	DCHAR		= 0, 
	DSHORT		= 1, 
	DINTEGER	= 2, 
	DFLOAT		= 3, 
	DDOUBLE		= 5
}; // memory/language data type

typedef char Byte;

class DataType {
private:
	DATA_TYPE _type;

public:
	DataType(DATA_TYPE type): _type(type) { }

	unsigned int dataSize() {
		switch (_type) {
		case DINTEGER:
			return sizeof(int);
			break;
		case DFLOAT:
			return sizeof(float);
			break;
		case DDOUBLE:
			return sizeof(double);
			break;
		default:
			return 0;
		}
	}

	// convert the pointed data to specific type
	template<class Type>	
	Type getValue(Byte *p) {
		switch (_type) {
		case DINTEGER:
			return *((int*)p);
			break;
		case DFLOAT:
			return *((float*)p);
			break;
		case DDOUBLE:
			return *((double*)p);
			break;
		default:
			return 0;
		}
	}
};

#endif