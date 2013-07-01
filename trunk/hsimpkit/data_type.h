/*
 *  A Class That Returns the Value the Given Pointer Points to
 *  Based on the Given Value Type
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

// A Class That Returns the Value the Given Pointer Points to
// Based on the Given Value Type
class DataType {
private:
	DATA_TYPE _type;

public:
	DataType(DATA_TYPE type): _type(type) { }

	unsigned int dataSize() {
		switch (_type) {
		case DINTEGER:
			return sizeof(int);
		case DFLOAT:
			return sizeof(float);
		case DDOUBLE:
			return sizeof(double);
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
		case DFLOAT:
			return *((float*)p);
		case DDOUBLE:
			return *((double*)p);
		default:
			return 0;
		}
	}
};

#endif