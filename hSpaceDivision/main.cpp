/*
 *	main function and cmdline process
 *
 *	author: ht
 */

#include "getopt.h"
#include <string>
#include <iostream>
#include "spatial_division.h"
#include "vertex_cluster.h"
#include "trivial.h"
#include "htime.h"

using std::string;
using std::cerr;
using std::cout;
using std::endl;

int target;
char *infilename;
char outfilename[200];

static char *options = "t:h";

static char *usage_string = 
"-t <n>\ttarget vertices of the simplified mesh\n"
"-h\tprint help\n"
"\n";

static void print_usage()
{
	//std::cerr << std::endl;
	cerr << endl << "usage: [execname] <options> [filename]" << endl;
	cerr << endl
		<< "available options:" << endl
		<< usage_string << endl;
}

static void usage_error(char *msg = NULL)
{
	if( msg )  cerr << "#error: " << msg << endl;
	print_usage();
	exit(1);
}

void process_cmdline(int argc, char **argv)
{
	int opt, ival;

	getopt_init();
	while( (opt = getopt(argc, argv, options)) != EOF )
	{
		// set parameters for spatial partition
		switch( opt )
		{
		case 't':
			ival = atoi(optarg);
			if( ival <= 0 )
				usage_error("target count should be positive\n");
			else target = ival;
			break;

		case 'h':  print_usage(); exit(0); break;

		default:   usage_error("malformed command line."); break;
		}
	}

	if (optind >= argc)
	{
		cerr << "#error: no file name input" << endl;
		exit(1);
	}

	string _sval(argv[optind]);
	if(_sval.substr( _sval.find_last_of('.') + 1 ).compare( "ply" ) != 0)
	{
		usage_error("file format not ply");
	}
	// set input file name
	infilename = argv[optind];
	trimExtAndAppend(infilename, outfilename, "_sdsimp.ply");
}

int main(int argc, char** argv)
{
	process_cmdline(argc, argv);

	HTime htime;

	HSpatialDivision sd;
	if (sd.readPly(infilename) == false) 
		return 1;
	cout << "\tread file time: " << htime.printElapseSec() << endl;

	if (sd.divide(target) == false) 
		return 1;
	cout << "\tsimplification time: " << htime.printElapseSec() << endl;

	if (sd.toPly(outfilename) == false) 
		return 1;

	sd.clear();

	return 0;
}

