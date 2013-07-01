/*
 *  Child Config Operations for Octrees 
 *
 *  Author: Ht
 *  Email : waytofall916 at gmail dot com
 *
 *  Copyright (C) Ht. All rights reserved.
 */

#ifndef _ADAPISO_CHILD_CONFIG_
#define _ADAPISO_CHILD_CONFIG_

/*********************************************
 the child split count in X, Y, Z dimension is
 stored in the 0-1, 2-3, 4-5 digits of the 
 child_config field. if it has no child, the 
 field is set to 0

 the two binary codes for each dimension mean:
 00: NONE
 01: LEFT OF THE NODE OCCUPIED
 10: RIGHT OF THE NODE OCCUPIED
 11: NODE IS WHOLLY OCCUPIED
**********************************************/

const unsigned char LEFT_OCCUPIED =  0x1;
const unsigned char RIGHT_OCCUPIED = 0x2;
const unsigned char WHOLE_OCCUPIED = 0x3;

__forceinline__ __device__
void setBit(unsigned char &config, const unsigned char &bit) {
	config |= (1<<bit);
}

__forceinline__ __device__
void clearBit(unsigned char &config, const unsigned char &bit) {
	config &= ~(1<<bit);
}

__forceinline__ __device__
void childConfigSetXLeft(unsigned char &config) {
	setBit(config, 0);
	clearBit(config, 1);
}

__forceinline__ __device__
void childConfigSetXRight(unsigned char &config) {
	clearBit(config, 0);
	setBit(config, 1);
}

__forceinline__ __device__
void childConfigSetXWhole(unsigned char &config) {
	setBit(config, 0);
	setBit(config, 1);
}

__forceinline__ __device__
void childConfigSetYLeft(unsigned char &config) {
	setBit(config, 2);
	clearBit(config, 3);
}

__forceinline__ __device__
void childConfigSetYRight(unsigned char &config) {
	clearBit(config, 2);
	setBit(config, 3);
}

__forceinline__ __device__
void childConfigSetYWhole(unsigned char &config) {
	setBit(config, 2);
	setBit(config, 3);
}

__forceinline__ __device__
void childConfigSetZLeft(unsigned char &config) {
	setBit(config, 4);
	clearBit(config, 5);
}

__forceinline__ __device__
void childConfigSetZRight(unsigned char &config) {
	clearBit(config, 4);
	setBit(config, 5);
}

__forceinline__ __device__
void childConfigSetZWhole(unsigned char &config) {
	setBit(config, 4);
	setBit(config, 5);
}

__forceinline__ __device__
void getXConfig(const unsigned char &config, unsigned char &x_config) {
	x_config = config & 0x03;
}

__forceinline__ __device__
void getYConfig(const unsigned char &config, unsigned char &y_config) {
	y_config = config & (0x03 << 2);
	y_config >>= 2;
}

__forceinline__ __device__
void getZConfig(const unsigned char &config, unsigned char &z_config) {
	z_config = config & (0x03 << 4);
	z_config >>= 4;
}

__forceinline__ __device__
void configToCount(unsigned char &config) {
	if (config == LEFT_OCCUPIED || config == RIGHT_OCCUPIED) 
		config = 1;
	else if (config == WHOLE_OCCUPIED)
		config = 2;
	else
		config = 0;
}

__forceinline__ __device__
void getChildStartCount(const unsigned char &config, unsigned char &start, unsigned char &count) {
	if (config == LEFT_OCCUPIED) {
		start =  0;
		count = 1;
	} else if (config == RIGHT_OCCUPIED) {
		start = 1;
		count = 1;
	} else {
		start = 0;
		count = 2;
	}
}

__forceinline__ __device__
void getChildCount(const unsigned char &config, unsigned char &count) {
	if (config == LEFT_OCCUPIED || config == RIGHT_OCCUPIED) {
		count = 1;
	} else {
		count = 2;
	}
}

#endif //_ADAPISO_CHILD_CONFIG_