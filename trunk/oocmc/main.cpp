#include <iostream>
#include <fstream>

#include "getopt.h"
#include "patching_simp.h"
#include "trivial.h"


using std::ofstream;
using std::fstream;
using std::cout;
using std::endl;

static double	iso;
static char*	filename;
static uint		x_div = 2, y_div = 2, z_div = 2;
static uint		target;

static bool		simplify = false;

static char		*options = "i:t:x:y:z:h";

static char *usage_string = 
"\t-i <n>\tiso value\n"
"\t-t <n>\ttarget verts of simplified generated models, simplification is chosen if this parameter is set\n"
"\t-x <n>\tx division for the partition grid\n"
"\t-y <n>\ty division for the partition grid\n"
"\t-z <n>\tz division for the partition grid\n"
"\t-h\tprint help\n";

static void print_usage()
{
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

static void process_cmdline(int argc, char **argv)
{
	int opt, ival;
	double fval;

	getopt_init();
	while( (opt = getopt(argc, argv, options)) != EOF )
	{
		switch( opt )
		{
		case 'i':
			fval = atof(optarg);
			iso = fval;
			break;

		case 't':
			ival = atoi(optarg);
			if( ival <= 0 )
				usage_error("target count should be positive\n");
			else {
				target = ival;
				simplify = true;
			}
			break;

		case 'x':
			ival = atoi(optarg);
			if( ival <= 0 )
				usage_error("x division should be positive\n");
			else x_div = ival;
			break;

		case 'y':
			ival = atoi(optarg);
			if( ival <= 0 )
				usage_error("y division should be positive\n");
			else y_div = ival;
			break;

		case 'z':
			ival = atoi(optarg);
			if( ival <= 0 )
				usage_error("z division should be positive\n");
			else z_div = ival;
			break;

		case 'h':
			print_usage();
			break;
			
		default:   
			usage_error("command line arguments error"); 
			break;
		}
	}

	if (optind >= argc) 
		usage_error("no file name input");

	string _sval(argv[optind]);
	if(_sval.substr(_sval.find_last_of('.') + 1).compare("dat") != 0)
		usage_error("file format not .dat");

	// set input file name
	filename = argv[optind];
}

extern int psimp_entry(char *filename, uint target, 
	uint x_div, uint y_div, uint z_div, bool binary);

int main(int argc, char** argv)
{
	filename = "d:/bunny/bun_zipper.ply";

	process_cmdline(argc, argv);



	return psimp_entry(filename, target, x_div, y_div, z_div, true);
}
