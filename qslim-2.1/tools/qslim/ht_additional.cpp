#include <iostream>
#include <fstream>
#include "stdmix.h"
#include "MxStdModel.h"
#include "qslim.h"

using std::ofstream;
using std::endl;

void output_ply(MxStdModel *m, char *filename)
{
	ofstream fout(filename);
	int vert_count = 0, face_count = 0;
	int i;

	for(i = 0; i < m->vert_count(); i ++)
		if( m->vertex_is_valid(i) )
			vert_count ++;

	for(i = 0; i < m->face_count(); i ++)
		if( m->face_is_valid(i) )
			face_count ++;

	fout << "ply" << endl;
	fout << "format ascii 1.0" << endl;
	fout << "comment Generated from MxStdModel model by QSlim" << endl;

	fout << "element vertex " << vert_count << endl;
	fout << "property float x" << endl;
	fout << "property float y" << endl;
	fout << "property float z" << endl;
	fout << "element face " << face_count << endl;
	fout << "property list uchar int vertex_indices" << endl;
	fout << "end_header" << endl;

	for(i = 0; i < m->vert_count(); i ++)
		if( m->vertex_is_valid(i) )
			fout << m->vertex(i)[0] << " "
			<< m->vertex(i)[1] << " "
			<< m->vertex(i)[2] << endl;

	for(i = 0; i < m->face_count(); i ++)
		if( m->face_is_valid(i) )
			fout << "3 "
			<< m->face(i)[0] << " "
			<< m->face(i)[1] << " "
			<< m->face(i)[2] << endl;
}
