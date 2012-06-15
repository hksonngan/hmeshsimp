/*
	main function and cmdline process
	for oocsimp, author: ht
*/

#include "getopt.h"
#include <string>
#include "oocsimp.h"
#include "global_var.h"
#include <iostream>
#include "vertex_cluster.h"

using std::string;
using std::cerr;
using std::cout;
using std::endl;

static char *options = "X:Y:Z:r:c:h";

static char *usage_string = 
"-X <n>\tX partitions of the grid\n"
"-Y <n>\tY partitions of the grid\n"
"-Z <n>\tZ partitions of the grid\n"
"-r\trepresentative vertex calculating policy\n"
"\t\tq: quadric metrics(default) m: mean vertex\n"
"				q: quadric metrics(default) m: mean vertex\n"
"-c\tcache size, 500000 default"
"-h\tprint help\n"
"\n";

/*
usage message:
"-X <n>	X partitions of the grid\n"
"-Y <n>	Y partitions of the grid\n"
"-Z <n>	Z partitions of the grid\n"
"-r		representative vertex calculating policy\n"
"				q: quadric metrics(default) m: mean vertex\n"
"-c		cache size, 500000 default"
"-h		print help\n"
"\n";
*/

static void print_usage()
{
	//std::cerr << std::endl;
	//slim_print_banner(cerr);
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
		// set parameters for g_oocsimp
		switch( opt )
		{
		case 'X':
			ival = atoi(optarg);
			if( ival <= 0 )
				usage_error("X partitions should be positive\n");
			else  g_oocsimp.x_partition = ival;
			break;

		case 'Y':
			ival = atoi(optarg);
			if( ival <= 0 )
				usage_error("Y partitions should be positive\n");
			else  g_oocsimp.y_partition = ival;
			break;

		case 'Z':
			ival = atoi(optarg);
			if( ival <= 0 )
				usage_error("Z partitions should be positive\n");
			else  g_oocsimp.z_partition = ival;
			break;

		case 'r':
			if (strcmp(optarg, "m") == 0) 
				g_oocsimp.rcalc_policy = MEAN_VERTEX;
			else 
				g_oocsimp.rcalc_policy = QEM_INV;
			break;

		case 'c':
			ival = atoi(optarg);
			if( ival <= 0 )
				usage_error("cache size should be positive\n");
			else  g_oocsimp.cache_size = ival;
			break;


		//case 'W':
		//	ival = atoi(optarg);
		//	if( ival<MX_WEIGHT_UNIFORM || ival>MX_WEIGHT_RAWNORMALS )
		//		usage_error("Illegal weighting policy.");
		//	else weighting_policy = ival;
		//	break;

		//case 'M':
		//	if( !select_output_format(optarg) )
		//		usage_error("Unknown output format selected.");
		//	break;

		//case 'B':  boundary_weight = atof(optarg); break;
		//case 't':  face_target = atoi(optarg); break;
		//case 'F':  will_use_fslim = true; break;
		//case 'o':  output_filename = optarg; break;
		//case 'I':  defer_file_inclusion(optarg); break;
		//case 'm':  meshing_penalty = atof(optarg); break;
		//case 'c':  compactness_ratio = atof(optarg); break;
		//case 'r':  will_record_history = true; break;
		//case 'j':  will_join_only = true; break;
		//case 'q':  be_quiet = true; break;
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
	g_oocsimp.infilename = argv[optind];
}

#include "vertex_cluster.h"

int main (int argc, char **argv)
{
	//std::cout << sizeof(TriSoup) << std::endl;
	//std::cout << sizeof(unsigned long) << std::endl;
	//HFaceIndexHash hash;
	//std::cout << hash(HFaceIndex(HTripleIndex(1, 2, 3), HTripleIndex(1, 2, 4), HTripleIndex(1, 2, 5))) << std::endl;
	//std::cout << hash(HFaceIndex(HTripleIndex(100, 2, 3), HTripleIndex(100, 2, 4), HTripleIndex(100, 2, 5))) << std::endl;

	process_cmdline(argc, argv);
	g_oocsimp.oocsimp();
}