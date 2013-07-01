/*
 *  Get the Option Parameter from Command Line
 *    - Dragged from the web
 *    - A similar implementation as in <unistd.h> of UNIX
 *
 *  Author: Ht
 *  Email : waytofall916 at gmail dot com
 *
 *  Copyright (C) Ht. All rights reserved.
 */

#ifndef GETOPT_H_
#define GETOPT_H_


#ifdef __cplusplus
extern "C" {
#endif

extern char* optarg;
extern int optind;
extern int opterr;
extern int optopt;

int getopt(int argc, char** argv, char* optstr);
void getopt_init();

#ifdef __cplusplus
}
#endif


#endif
