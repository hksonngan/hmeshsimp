/*
 *	A timer
 *
 *  Author: Ht
 *  Email : waytofall916@gmail.com
 *
 *  Copyright (C) Ht-waytofall. All rights reserved.
 *	
 *  This file is part of hmeshsimp.
 *
 *  hmeshsimp is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  hmeshsimp is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with hmeshsimp.  If not, see <http://www.gnu.org/licenses/>.
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

	void setStartPoint() {
		time_stamp = clock();
	}

	void setEndPoint() {
		sprintf(buf, "%f sec", (clock() - (float)time_stamp) / CLOCKS_PER_SEC);
	}

	char* getElapseStr() {
		return buf;
	}

	const char* printElapseSec() {
		setEndPoint();
		return buf;
	}

private:
	clock_t time_stamp;
	char buf[BUF_SIZE];
};

#endif