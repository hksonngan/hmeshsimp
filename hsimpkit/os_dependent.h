/*
 *  This file defines some operating system dependent
 *  functions invoked by some of the os independent codes, 
 *  which should be written separately on different operating
 *  systems.
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

#ifndef __OS_DEPENDENT__
#define __OS_DEPENDENT__

// create a file directory using the os specific kernel call.
// returns true is the directory has already existed
extern bool hCreateDir(char *dir_path);

// '\' or '/'
inline extern char* hPathSeperator();

#endif //__OS_DEPENDENT__