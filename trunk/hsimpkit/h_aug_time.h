/*
 *  Augmented timing
 *
 *  Author: Ht
 *  Email : waytofall916 at gmail dot com
 *
 *  Copyright (C) Ht-waytofall. All rights reserved.
 */


#ifndef __H_AUG_TIME__
#define __H_AUG_TIME__

#include <stdio.h>
#include <iostream>
#include "gfx/gfx.h"

using std::ostream;

#define BUF_SIZE 500

class HAugTime {
public:
	HAugTime() {
		start_time = get_cpu_time();
		time_inverval = 0;
	}

	void setCheckPoint() {
		start_time = get_cpu_time();
	}
	void start() {
		start_time = get_cpu_time();
	}

	void setStartPoint() {
		start_time = get_cpu_time();
	}
	void end() {
		double end_time = get_cpu_time();
		time_inverval = end_time - start_time;
		start_time = end_time;
	}

	void setEndPoint() {
		double end_time = get_cpu_time();
		time_inverval = end_time - start_time;
		start_time = end_time;
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
	double start_time, time_inverval;
	char buf[BUF_SIZE];
};

inline ostream& operator << (ostream& out, HAugTime htime) {

	out << htime.getElapseStr();
	return out;
}

#endif //__H_AUG_TIME__