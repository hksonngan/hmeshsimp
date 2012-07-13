/*
 *  iterative edge collapse
 *
 *  author: ht
 *  email : waytofall916@gmail.com
 */

#include "ecol_iterative_quadric.h"
#include "trivial.h"
#include "getopt.h"

#include <iostream>
#include <fstream>
#include <string>

using std::cerr;
using std::cout;
using std::endl;
using std::string;
using std::ofstream;

#define BUF_SIZE 1000

char*	infilename;
char	outfilename[BUF_SIZE];
int		target = -1;
double	boundary_weight = INIT_BOUND_WEIGHT;

static char *options = "t:b:h";

static char *usage_string = 
"\t-t <n>\ttarget faces of the simplified mesh\n"
"\t-b <n>\tboundary weight\n"
"\t-h\tprint help\n";

static void print_usage()
{
	//std::cerr << std::endl;
	cerr << endl << "\tusage: [execname] <options> [filename]" << endl;
	cerr << endl
		<< "\tavailable options:" << endl
		<< usage_string << endl;
}

static void usage_error(char *msg = NULL)
{
	if( msg )  cerr << "\t#ERROR: " << msg << endl;
	print_usage();
	exit(1);
}

void process_cmdline(int argc, char **argv)
{
	int opt, ival;
	double fval;

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

		case 'b':
			fval = atof(optarg);
			if( fval <= 0 )
				usage_error("boundary weight should be positive\n");
			else boundary_weight = fval;
			break;

		case 'h':  print_usage(); exit(0); break;

		default:   usage_error("command line arguments error"); break;
		}
	}

	if (optind >= argc) 
		usage_error("no file name input");

	string _sval(argv[optind]);
	if(_sval.substr( _sval.find_last_of('.') + 1 ).compare( "ply" ) != 0)
		usage_error("file format not ply");

	// set input file name
	infilename = argv[optind];
	trimExtAndAppend(infilename, outfilename, "_qesimp.ply");
}

int main(int argc, char** argv) {
	
	process_cmdline(argc, argv);

	QuadricEdgeCollapse qec;
	bool r;
	ofstream flog("ec.log", fstream::out | fstream::app);

	qec.boundary_weight = boundary_weight;

	r = qec.readPly(infilename);
	flog << qec.getInfo();
	cout << qec.getInfo();
	qec.clearInfo();
	if (!r)
		return 1;

	if (target ==  -1) 
		target = qec.faceCount() / 2;

	r = qec.targetFace(target);
	flog << qec.getInfo();
	cout << qec.getInfo();
	qec.clearInfo();
	if (!r)
		return 1;

	//qec.outputIds("ids");

	r = qec.writePly(outfilename);
	flog << qec.getInfo();
	cout << qec.getInfo();
	qec.clearInfo();
	if (!r)
		return 1;

	qec.totalTime();
	flog << qec.getInfo();
	cout << qec.getInfo();
	qec.clearInfo();

	return 0;
}