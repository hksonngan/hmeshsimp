/*
	a timer
	author: ht
	email : waytofall016@gmail.com
*/

#ifndef __H_TIME__
#define __H_TIME__

#include <time.h>
#include <stdio.h>

#define BUF_SIZE 50

class HTime
{
public:
	HTime() {
		time_stamp = clock();
	}

	void setCheckPoint() {
		time_stamp = clock();
	}

	const char* printElapseSec() {
		sprintf(buf, "%f sec", (clock() - (float)time_stamp) / CLOCKS_PER_SEC);
		return buf;
	}

private:
	clock_t time_stamp;
	char buf[BUF_SIZE];
};

#endif