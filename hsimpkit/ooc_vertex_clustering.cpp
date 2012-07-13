#include "ooc_vertex_clustering.h"
#include <iostream>
#include <fstream>
#include <string>
#include "ply/ply_inc.h"
#include "lru_cache.h"
#include "tri_soup_stream.h"
#include "h_time.h"

using std::cout;
using std::endl;

//#define _OUTPUT_TRI_SOUP
//#define _OUTPUT_VERT_TXT

static void print_error(char* e_msg)
{
	std::cerr << std::endl << std::endl << "#error: " << e_msg << std::endl;
	exit(0);
}

static void determinBoundBox(BoundBox& bound_box, Vertex vertex)
{
	if (bound_box.max_x < vertex.x) {
		bound_box.max_x = vertex.x;
	}
	if (bound_box.min_x > vertex.x) {
		bound_box.min_x = vertex.x;
	}
	if (bound_box.max_y < vertex.y) {
		bound_box.max_y = vertex.y;
	}
	if (bound_box.min_y > vertex.y) {
		bound_box.min_y = vertex.y;
	}
	if (bound_box.max_z < vertex.z) {
		bound_box.max_z = vertex.z;
	}
	if (bound_box.min_z > vertex.z) {
		bound_box.min_z = vertex.z;
	}
}

static void freePointersInFace(Face *face) {
	if (face->verts) {
		free(face->verts);
	}
	if (face->other_props) {
		free(face->other_props);
	}
}

/*
	the thesis of [Lindstrom 2000] needs the input file
	to be triangle soup, while the standard model input
	is mostly indexed mesh, so I need some techniques
	and investigation to try to first convert the indexed
	mesh form to triangle soup. converting it needs to
	randomly access the vertex, which means constantly
	irregular disk fetching, the way I solve this problem
	is by using a hashing cache for constant time fetching
	of cached unit while using LEAST RECENT USED strategy
	to pop out the existing cached content when the cache
	is full. please refer to the class VertexBinary for
	further information. Another approach for the transformation 
	is using external sorts [Lindstrom.Silva 2001].
	-- ht
*/

int HOOCSimp::toTriangleSoup()
{
	int i,j;
	PlyFile *ply;
	int nprops;
	int num_elems;
	PlyProperty **plist = NULL;
	char *elem_name;
	float version;
	Vertex vertex;
	Face face;

	/* a space for PlyFile is malloced in the function
	   it must be freed after using, as well as elist - houtao */
	ply = ply_open_for_reading(infilename, &nelems, &elist, &file_type, &version);
	if(ply == NULL) {
		std::cout << "#error: ply file open failed" << std::endl;
		return 0;
	}

	VertexBinary vertex_bin;
	string vertexbin_filename(infilename);
	vertexbin_filename = vertexbin_filename.substr(0, vertexbin_filename.find_last_of(".")) + "_vert.bin";
	if ( vertex_bin.openForWrite( vertexbin_filename.c_str() ) == 0 ) {
		std::cout << "#error: vertex binary file open failed while writing" << std::endl;
		return 0;
	}

	for (i = 0; i < nelems; i++) {

		/* get the description of the first element */
		elem_name = elist[i];
		/* the prop_list as well as the property in it and the char* name in the property is malloced - houtao */
		plist = ply_get_element_description (ply, elem_name, &num_elems, &nprops);

		if (equal_strings ("vertex", elem_name)) {

			nverts = num_elems;
			#ifdef _OUTPUT_VERT_TXT 
			std::ofstream vert_out ("vert.txt");
			#endif

			/* set up for getting vertex elements */

			ply_get_property (ply, elem_name, &vert_props[0]);
			ply_get_property (ply, elem_name, &vert_props[1]);
			ply_get_property (ply, elem_name, &vert_props[2]);

			/* the function malloc an OtherProperty and return it
			   but luckily that vert_other is a global variable
			   and can be freed in clean_ply() - houtao */
			vert_other = ply_get_other_properties (ply, elem_name,
				offsetof(Vertex, other_props));

			/* get the first vertex and set the bounding box */
			vertex.other_props = NULL;
			ply_get_element (ply, (void *) &vertex);

			bound_box.max_x = vertex.x;
			bound_box.min_x = vertex.x;
			bound_box.max_y = vertex.y;
			bound_box.min_y = vertex.y;
			bound_box.max_z = vertex.z;
			bound_box.min_z = vertex.z;

			if (vertex_bin.writeVertexFloat(vertex.x, vertex.y, vertex.z) == 0) {
				std::cout << "#error: vertex binary file write failed" << std::endl;
				return 0;
			}

			#ifdef _OUTPUT_VERT_TXT 
			vert_out << vertex.x << " " << vertex.y << " " << vertex.z << std::endl;
			#endif

			/* grab all the vertex elements */
			for (j = 1; j < num_elems; j ++) {
				vertex.other_props = NULL;
				ply_get_element (ply, (void *) &vertex);

				determinBoundBox(bound_box, vertex);

				if (vertex_bin.writeVertexFloat(vertex.x, vertex.y, vertex.z) == 0) {
					std::cout << "#error: vertex binary file write failed" << std::endl;
					return 0;
				}

				#ifdef _OUTPUT_VERT_TXT 
				vert_out << vertex.x << " " << vertex.y << " " << vertex.z << std::endl;
				#endif

				if (vertex.other_props) {
					free(vertex.other_props);
				}
				
			}

			vertex_bin.closeWriteFile();

			#ifdef _OUTPUT_VERT_TXT 
			vert_out.close();
			#endif
		}
		else if (equal_strings ("face", elem_name)) {

			/* create a list to hold all the face elements */
			nfaces = num_elems;
			Vertex v1, v2, v3;
			vertex_bin.initCache(cache_size);

			#ifdef _OUTPUT_TRI_SOUP
			std::ofstream tris_t ("triangle_soup.txt");
			#endif

			if ( vertex_bin.openForRead( vertexbin_filename.c_str() ) == 0 ) {
				std::cout << "#error: vertex binary file open failed" << std::endl;
				return 0;
			}

			std::string str_tris_filename = infilename;
			str_tris_filename = str_tris_filename.substr(0, str_tris_filename.find_last_of(".")) + ".tris";
			if (tris_filename) {
				delete[] tris_filename;
			}
			tris_filename = new char[str_tris_filename.size() + 1];
			memcpy(tris_filename, str_tris_filename.c_str(), str_tris_filename.size());
			tris_filename[str_tris_filename.size()] = '\0';

			TriSoupStream tris;
			tris.setBoundBox(bound_box.max_x, bound_box.min_x, bound_box.max_y, bound_box.min_y,
				bound_box.max_z, bound_box.min_z);
			if ( tris.openForWrite( tris_filename ) == 0 ) {
				std::cout << "#error: triangle soup file open failed while writing" << std::endl;
				return 0;
			}

			/* set up for getting face elements */

			ply_get_property (ply, elem_name, &face_props[0]);
			face_other = ply_get_other_properties (ply, elem_name,
				offsetof(Face,other_props));

			/* grab all the face elements */
			for (j = 0; j < num_elems; j++) {
				face.nverts = NULL;
				face.other_props = NULL;
				ply_get_element (ply, (void *) &face);
				
				//vertex_bin.writeCacheDebug();

				if (face.nverts != 3) {
					std::cerr << "#error: non triangle in input file" << std::endl;
					freePointersInFace(&face);
					continue;
				}

				/* read from cached vertex binary file */
				
				if (vertex_bin.indexedRead(face.verts[0]) == 0) {
					std::cerr << "#error: vertex binary file read failed" << std::endl;
					return 0;
				}
				v1.x = vertex_bin.getXFloat();
				v1.y = vertex_bin.getYFloat();
				v1.z = vertex_bin.getZFloat();
				if (vertex_bin.indexedRead(face.verts[1]) == 0) {
					std::cerr << "#error: vertex binary file read failed" << std::endl;
					return 0;
				}
				v2.x = vertex_bin.getXFloat();
				v2.y = vertex_bin.getYFloat();
				v2.z = vertex_bin.getZFloat();
				if (vertex_bin.indexedRead(face.verts[2]) == 0) {
					std::cerr << "#error: vertex binary file read failed" << std::endl;
					return 0;
				}
				v3.x = vertex_bin.getXFloat();
				v3.y = vertex_bin.getYFloat();
				v3.z = vertex_bin.getZFloat();

				/* write to '.tris' file */
				if(tris.writeFloat(v1.x, v1.y, v1.z, v2.x, v2.y, v2.z, v3.x, v3.y, v3.z) == 0) {
					std::cerr << "#error: write triangle soup file read failed" << std::endl;
					return 0;
				}

				#ifdef _OUTPUT_TRI_SOUP
				tris_t << v1.x << " " << v1.y << " " << v1.z << " " 
					<< v2.x << " " << v2.y << " " << v2.z << " " 
					<< v3.x << " " << v3.y << " " << v3.z << std::endl ;
				#endif

				freePointersInFace(&face);
			}

			vertex_bin.closeReadFile();
			tris.closeForWrite();

			#ifdef _OUTPUT_TRI_SOUP
			tris_t.close();
			#endif

			// write statistics
			std::cout << std::endl << "\t#info:" << std::endl
				<< "\ttriangle soup convert successfully" << std::endl
				<< "\tfile name: " << infilename << std::endl
				<< "\tvertex count: " << nverts << " face count: " << nfaces << std::endl
				<< "\tcache size: " << cache_size << endl
				<< "\tvertext file read count: " << vertex_bin.read_count << std::endl
				<< "\thit count: " << vertex_bin.hit_count << std::endl
				<< "\thit rate: %" << (int)((float)vertex_bin.hit_count / vertex_bin.read_count * 100) << std::endl;
		}
		else
			other_elements = ply_get_other_element (ply, elem_name, num_elems);

		/* free plist and its memory space in case of memory leaking */
		if (plist) {
			for (int i = 0; i < nprops; i ++) {
				if (plist[i]) {
					if (plist[i]->name) {
						free(plist[i]->name); 
					}
					free(plist[i]);
				}
			}
			free(plist);
		}

	}

	comments = ply_get_comments (ply, &num_comments);
	obj_info = ply_get_obj_info (ply, &num_obj_info);

	/* the ply variable is freed in the function - houtao */
	ply_close (ply);

	return 1;
}

int HOOCSimp::oocsimp()
{
	HTime htime;

	if (toTriangleSoup() == 0) {
		return 0;
	}

	cout << "\trunning time: " << htime.printElapseSec() << endl;

	std::string s_ply_name = infilename;
	s_ply_name = s_ply_name.substr(0, s_ply_name.find_last_of(".")) + "_simp.ply";
	char *cstr_ply_name = new char[s_ply_name.size() + 1];
	memcpy(cstr_ply_name, s_ply_name.c_str(), s_ply_name.size());
	cstr_ply_name[s_ply_name.size()] = '\0';

	std::cout << std::endl;

	htime.setCheckPoint();
	if (HOOCVertexClustering().run(x_partition, y_partition, z_partition, rcalc_policy, tris_filename, cstr_ply_name) == false) {
		std::cerr << "#error: error occurred when performing out-of-core vertex clustering" << std::endl;
		return 0;
	}
	
	cout << "\trunning time: " << htime.printElapseSec() << endl;

	delete[] cstr_ply_name;

	return 1;
}


bool HOOCVertexClustering::run(int x_partition, int y_partition, int z_partition, RepCalcPolicy p,
	char* inputfilename, char* outputfilename)
{
	TriSoupStream sstream;
	if (sstream.openForRead(inputfilename) == 0) {
		std::cerr << "#error: open triangle soup file failed" << std::endl;
		return false;
	}

	HVertexClusterSimp vcsimp;
	HSoupTriangle soup_tri;

	vcsimp.setBoundBox(sstream.getMaxX(), sstream.getMinX(), sstream.getMaxY(), 
		sstream.getMinY(), sstream.getMaxZ(), sstream.getMinZ());
	vcsimp.create(x_partition, y_partition, z_partition, p);

	while (sstream.readNext())
	{
		// retrieve the triangle soup
		soup_tri.v1.Set(sstream.getFloat(0, 0), sstream.getFloat(0, 1), sstream.getFloat(0, 2));
		soup_tri.v2.Set(sstream.getFloat(1, 0), sstream.getFloat(1, 1), sstream.getFloat(1, 2));
		soup_tri.v3.Set(sstream.getFloat(2, 0), sstream.getFloat(2, 1), sstream.getFloat(2, 2));

		vcsimp.addSoupTriangle(soup_tri);
	}

	if (vcsimp.writeToPly(outputfilename) == false) {
		std::cerr << "#error: write to ply file failed" << std::endl;
		return false;
	}

	return true;
}