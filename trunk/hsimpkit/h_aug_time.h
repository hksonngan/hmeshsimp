/*
 *  Augmented timing
 *
 *  Author: Ht
 *  Email : waytofall916@gmail.com
 *
 *  Copyright (C) Ht-waytofall. All rights reserved.
 */


#ifndef __H_AUG_TIME__
#define __H_AUG_TIME__

#include <stdio.h>
#include "gfx/gfx.h"

#define BUF_SIZE 500

class HAugTime {
public:
	HAugTime() {
		time_inverval = get_cpu_time();
	}

	void setCheckPoint() {
		time_inverval = get_cpu_time();
	}

	void setStartPoint() {
		time_inverval = get_cpu_time();
	}

	void setEndPoint() {
		time_inverval = get_cpu_time() - time_inverval;
	}

	const char* getElapseStr() {
		sprintf(buf, "%f sec", time_inverval);
		return buf;
	}

	const char* printElapseSec() {
		setEndPoint();
		return getElapseStr();
	}

	HAugTime& operator = (const HAugTime &t) {
		time_inverval = t.time_inverval;
		return *this;
	}

	HAugTime& operator += (const HAugTime &t) {
		time_inverval += t.time_inverval;
		return *this;
	}

	HAugTime operator + (const HAugTime &t) {
		HAugTime _t;
		_t.time_inverval = time_inverval + t.time_inverval;
		return _t;
	}

private:
	double time_inverval;
	char buf[BUF_SIZE];
};

#endif //__H_AUG_TIME__