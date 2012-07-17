#include "ply_stream.h"
#include "ply.h"
#include "trivial.h"

PlyStream::PlyStream()
{
	ply = NULL;
	vCount = 0;
	fCount = 0;
	vertexElem = NULL;
	faceElem = NULL;
}

bool PlyStream::openForRead(const char *filename)
{
	int file_type;
	int nelems;
	char **elist;
	float version;
	int nprops;
	int num_elems;
	PlyProperty **plist = NULL;
	char *elem_name;
	PlyOtherProp *vert_other, *face_other;

	int i;

	/* a space for PlyFile is malloced in the function
	   it must be freed after using, as well as elist - houtao */
	ply = ply_open_for_reading(filename, &nelems, &elist, &file_type, &version);
	if(ply == NULL) {
		cerr << "#error: ply file open failed" << endl;
		return false;
	}

	/* check the endian mode */
	SYSTEM_ENDIAN_MODE = getSystemEndianMode();
	if (ply->file_type == PLY_BINARY_BE/* binary PLY file, big endian */) {
		FILE_ENDIAN_MODE = H_BIG_ENDIAN;
	}
	else if (ply->file_type == PLY_BINARY_LE/* binary PLY file, little endian */) {
		FILE_ENDIAN_MODE = H_LITTLE_ENDIAN;
	}

	for (i = 0; i < nelems; i++) {

		/* get the description of the first element */
		elem_name = elist[i];
		/* the prop_list as well as the property in it and the char* name in the property is malloced - houtao */
		plist = ply_get_element_description (ply, elem_name, &num_elems, &nprops);

		if (equal_strings ("vertex", elem_name)) {

			ply_get_property (ply, elem_name, &vert_props[0]);
			ply_get_property (ply, elem_name, &vert_props[1]);
			ply_get_property (ply, elem_name, &vert_props[2]);
			/* the function malloc an OtherProperty and return it
			   but luckily that vert_other is a global variable
		       and can be freed in clean_ply() - houtao */
			vert_other = ply_get_other_properties (ply, elem_name, offsetof(Vertex, other_props));

			vertexElem = find_element(ply, "vertex");
			vCount = num_elems;
		}
		else if (equal_strings ("face", elem_name)) {

			ply_get_property (ply, elem_name, &face_props[0]);
			face_other = ply_get_other_properties (ply, elem_name,
				offsetof(Face,other_props));

			faceElem = find_element(ply, "face");
			fCount = num_elems;
		}

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

	readVCount = 0;
	readFCount = 0;

	return true;
}

bool PlyStream::close()
{
	ply_close(ply);

	return true;
}
