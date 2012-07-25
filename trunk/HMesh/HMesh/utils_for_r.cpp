/*
	uitility functions for r[RESEARCH]
	author: ht
*/

//#include "hGlWidget.h"
#include "utils_for_r.h"
#include "tri_soup.h"
#include "ply/ply_inc.h"
#include <iostream>

using namespace std;

inline bool compVert(Vertex v, TripleFloat t)
{
	if (v.x != t.x || v.y != t.y || v.z != t.z)
	{
		return false;
	}
	return true;
}

void comparePlyTris(char* plyfilename, const char* trisfilename)
{
	TriangleSoupContainer tris_container;

	clean_ply();
	tris_container.clear();
	ply_read_file(plyfilename);

	tris_container.init();
	if (tris_container.read(trisfilename) == false)
	{
		cerr << "#error: read triangle soup file failed" << endl;
		tris_container.clear();
		return;
	}

	if (nfaces != tris_container.count())
	{
		cout << "#difference: count different" << endl;
		return;
	}

	Vertex v;
	Face f;
	bool tFlag = true;
	bool lFlage;

	for(int i = 0; i < nfaces; i ++)
	{
		lFlage = true;
		f = flist[i];

		if( f.nverts == 3 )
		{
			if (compVert(vlist[f.verts[0]], tris_container(i).vert1) == false)
			{
				lFlage = false;
			}

			if (compVert(vlist[f.verts[1]], tris_container(i).vert2) == false)
			{
				lFlage = false;
			}

			if (compVert(vlist[f.verts[2]], tris_container(i).vert3) == false)
			{
				lFlage = false;
			}
			
			if (lFlage == false)
			{
				cout << "difference, no." << i << endl
					<< "ply:" << endl
					<< vlist[f.verts[0]].x << " " << vlist[f.verts[0]].y << " " << vlist[f.verts[0]].z << endl
					<< vlist[f.verts[1]].x << " " << vlist[f.verts[1]].y << " " << vlist[f.verts[1]].z << endl
					<< vlist[f.verts[2]].x << " " << vlist[f.verts[2]].y << " " << vlist[f.verts[2]].z << endl
					<< "tri soup:" << endl
					<< tris_container(i).vert1.x << " " << tris_container(i).vert1.y << " " << tris_container(i).vert1.z << endl
					<< tris_container(i).vert2.x << " " << tris_container(i).vert2.y << " " << tris_container(i).vert2.z << endl
					<< tris_container(i).vert3.x << " " << tris_container(i).vert3.y << " " << tris_container(i).vert3.z << endl
					<< endl << endl;

				tFlag = false;
			}
		}
		else
		{
			cerr << "#error: no." << i << " poly non-triangle in ply models" << endl;
		}
	}

	if (tFlag)
	{
		cout << "#same: two files contains the same" << endl;
	}
}