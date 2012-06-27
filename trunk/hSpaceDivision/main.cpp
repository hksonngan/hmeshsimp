/*
 *	main function and cmdline process
 *
 *	author: ht
 */

#include "getopt.h"
#include <string>
#include <iostream>
//#include "spatial_division.h"
#include "spatial_division2.h"
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

static char *options = "t:m:h";

static char *usage_string = 
"\n"
"\t-t <n>\ttarget vertices of the simplified mesh\n\n"
"\t-m <n>\tminimum factor for normal variation.\n"
"\t\tthis value is bigger means that the area\n"
"\t\tcounts more importantly than normal variation.\n"
"\t\tdefault 0.5, must falls in [0, 1].\n\n"
"\t-h\tprint help\n\n";

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
	if( msg )  cerr << "#" << msg << endl;
	print_usage();
	exit(1);
}

void process_cmdline(int argc, char **argv)
{
	int opt, ival;
	float fval;

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

		case 'm':
			fval = atof(optarg);
			if( fval < 0 || fval > 1)
				usage_error("minimum normal variation factor should fall in [0, 1]\n");
			else HSDVertexCluster2::MINIMUM_NORMAL_VARI = fval;
			break;

		case 'h':  print_usage(); exit(0); break;

		default:   usage_error("command line arguments error"); break;
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

	//HTripleIndexHash hash;
	//cout << hash(HTripleIndex<Integer>(2342341, 2435345, 652652)) << endl;
	//cout << hash(HTripleIndex<Integer>(652652, 2435345, 2342341)) << endl;
	//cout << hash(HTripleIndex<Integer>(652652, 0, 2342341)) << endl;
	//cout << hash(HTripleIndex<Integer>(4, 1, 234241)) << endl;
	//return 0;

	HTime htime;

	HSpatialDivision2 sd;
	if (sd.readPly(infilename) == false) 
		return 1;

	if (sd.divide(target) == false) 
		return 1;

	if (sd.toPly(outfilename) == false) 
		return 1;		 

	sd.clear();

	return 0;
}

