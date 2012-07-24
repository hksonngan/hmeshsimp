#include <iostream>
#include <fstream>

#include "getopt.h"
#include "patching_simp.h"
#include "trivial.h"


using std::ofstream;
using std::fstream;
using std::cout;
using std::endl;

static char*	filename;
static uint		x_div = 2, y_div = 2, z_div = 2;
static uint		target = 20000;

static char		*options = "t:x:y:z:h";

static char *usage_string = 
"\t-t <n>\ttarget verts of the simplified mesh\n"
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
		// set parameters for spatial partition
		switch( opt )
		{
		case 't':
			ival = atoi(optarg);
			if( ival <= 0 )
				usage_error("target count should be positive\n");
			else target = ival;
			break;

		case 'x':
			ival = atof(optarg);
			if( ival <= 0 )
				usage_error("x division should be positive\n");
			else x_div = ival;
			break;

		case 'y':
			ival = atof(optarg);
			if( ival <= 0 )
				usage_error("y division should be positive\n");
			else y_div = ival;
			break;

		case 'z':
			ival = atof(optarg);
			if( ival <= 0 )
				usage_error("z division should be positive\n");
			else z_div = ival;
			break;

		default:   usage_error("command line arguments error"); break;
		}
	}

	if (optind >= argc) 
		usage_error("no file name input");

	string _sval(argv[optind]);
	if(_sval.substr(_sval.find_last_of('.') + 1).compare("ply") != 0)
		usage_error("file format not ply");

	// set input file name
	filename = argv[optind];
}

int main(int argc, char** argv)
{
	PatchingSimp psimp;
	ofstream flog("patchingsimp.log", fstream::app);
	bool ret;
	char tmp_dir[200];

	// F:/plys/bunny/bun_zipper.ply
	// F:/plys/happy_recon/happy_vrip.ply
	// F:/plys/xyzrgb_statuette.ply
	// F:/plys/lucy.ply

	// d:/bunny/bun_zipper.ply
	// d:/happy_recon/happy_vrip.ply
	// d:/hmeshsimp/patchingsimp/bun_zipper_patches/bun_zipper_psimp_bin.ply
	// d:/xyzrgb_statuette.ply
	// d:/lucy.ply

	filename = "F:/plys/lucy.ply";
	x_div = 2;
	y_div = 2;
	z_div = 2;
	target = 30000;

	process_cmdline(argc, argv);

	stringToCstr(getFilename(filename) + "_patches", tmp_dir);

	flog << endl << endl << 
		"\t###############################################" << endl 
		<< "\t" << getTime()
		<< "\t" << getExtFilename(filename) << endl;

	psimp.tmpBase(tmp_dir);
	ret = psimp.readPlyFirst(filename);
	if (!ret)
		goto termin;

	ret = psimp.readPlySecond(x_div, y_div, z_div);
	if (!ret)
		goto termin;

	//psimp.patchesToPly();
	//psimp.simplfiyPatchesToPly(target);

	ret = psimp.mergeSimpPly(target, true);
	if (!ret)
		goto termin;

	termin:
	flog << psimp.info();

	if (ret)
		return EXIT_SUCCESS;
	else
		return EXIT_FAILURE;
}
