/*
 *	read in ply files
 *  this file is more like a sample code for using Greg Turk's ply routines,
 *  however, there a lot more that behaves like this
 *	 -- ht
 */

/*

  Convert PLY models into SMF models.

  Michael Garland:  Oct 1996


  Based on a skeleton program
  (Greg Turk, August 1994)

*/

//#include <stdmix.h>
#include "ply/ply.h"
#include <iostream>
#include <fstream>
#include "ply_inc.h"


/* user's vertex and face definitions for a polygonal object */

//typedef struct Vertex {
//  float x,y,z;
//  void *other_props;       /* other properties */
//} Vertex;
//
//typedef struct Face {
//  unsigned char nverts;    /* number of vertex indices in list */
//  int *verts;              /* vertex index list */
//  void *other_props;       /* other properties */
//} Face;

char *elem_names[] = { /* list of the kinds of elements in the user's object */
  "vertex", "face"
};

PlyProperty vert_props[] = { /* list of property information for a vertex */
  {"x", PLY_FLOAT, PLY_FLOAT, offsetof(Vertex,x), 0, 0, 0, 0},
  {"y", PLY_FLOAT, PLY_FLOAT, offsetof(Vertex,y), 0, 0, 0, 0},
  {"z", PLY_FLOAT, PLY_FLOAT, offsetof(Vertex,z), 0, 0, 0, 0},
  {"r", PLY_FLOAT, PLY_FLOAT, offsetof(Vertex,r), 0, 0, 0, 0},
  {"g", PLY_FLOAT, PLY_FLOAT, offsetof(Vertex,g), 0, 0, 0, 0},
  {"b", PLY_FLOAT, PLY_FLOAT, offsetof(Vertex,b), 0, 0, 0, 0},
};

PlyProperty face_props[] = { /* list of property information for a vertex */
  {"vertex_indices", PLY_INT, PLY_INT, offsetof(Face,verts),
   1, PLY_UCHAR, PLY_UCHAR, offsetof(Face,nverts)},
  {"r", PLY_FLOAT, PLY_FLOAT, offsetof(Face,r), 0, 0, 0, 0},
  {"g", PLY_FLOAT, PLY_FLOAT, offsetof(Face,g), 0, 0, 0, 0},
  {"b", PLY_FLOAT, PLY_FLOAT, offsetof(Face,b), 0, 0, 0, 0},
};


/*** the PLY object ***/

/*static*/ int nverts,nfaces;
// data structure altered by houtao
// /*static*/ Vertex **vlist;
// /*static*/ Face **flist;
/*static*/ Vertex *vlist;
/*static*/ Face *flist;
/*static*/ int nelems;
/*static*/ char **elist;
/*static*/ PlyOtherElems *other_elements = NULL;
/*static*/ PlyOtherProp *vert_other,*face_other;
/*static*/ int num_comments;
/*static*/ char **comments;
/*static*/ int num_obj_info;
/*static*/ char **obj_info;
/*static*/ int file_type;


/******************************************************************************
Read in the PLY file from standard in.
******************************************************************************/

void ply_read_file(char* filename)
{
  int i,j;
  PlyFile *ply;
  int nprops;
  int num_elems;
  PlyProperty **plist = NULL;
  char *elem_name;
  float version;


  /*** Read in the original PLY object ***/

  //a space for PlyFile is malloced in the function
  //it must be freed after using, as well as elist - houtao
  ply = ply_open_for_reading(filename, &nelems, &elist, &file_type, &version);
  if(ply == NULL)
  {
    std::cout << "ply file open failed" << std::endl;
    return;
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
	// the prop_list as well as the property in it and the char* name in the property is malloced - houtao
    plist = ply_get_element_description (ply, elem_name, &num_elems, &nprops);

    if (equal_strings ("vertex", elem_name)) {

      /* create a vertex list to hold all the vertices */
	  // data structure altered by houtao
	  //vlist = (Vertex **) malloc (sizeof (Vertex *) * num_elems);
      vlist = (Vertex *) malloc (sizeof (Vertex) * num_elems);
      nverts = num_elems;

      /* set up for getting vertex elements */

      ply_get_property (ply, elem_name, &vert_props[0]);
      ply_get_property (ply, elem_name, &vert_props[1]);
      ply_get_property (ply, elem_name, &vert_props[2]);
	  //ply_get_property (ply, elem_name, &vert_props[3]);
	  //ply_get_property (ply, elem_name, &vert_props[4]);
	  //ply_get_property (ply, elem_name, &vert_props[5]);

	  // the function malloc an OtherProperty and return it
	  // but luckily that vert_other is a global variable
	  // and can be freed in clean_ply() - houtao
      vert_other = ply_get_other_properties (ply, elem_name,
                     offsetof(Vertex,other_props));

      /* grab all the vertex elements */
      for (j = 0; j < num_elems; j++) {
	    // modified by houtao
        //vlist[j] = (Vertex *) malloc (sizeof (Vertex));
		vlist[j].other_props = NULL;
		// modified by ht
        //ply_get_element (ply, (void *) &vlist[j]);
		ply_get_element (ply, reinterpret_cast<void*>(vlist+j));
      }
    }
    else if (equal_strings ("face", elem_name)) {

      /* create a list to hold all the face elements */
      flist = (Face *) malloc (sizeof (Face) * num_elems);
      nfaces = num_elems;

      /* set up for getting face elements */

      ply_get_property (ply, elem_name, &face_props[0]);
	  //ply_get_property (ply, elem_name, &face_props[1]);
	  //ply_get_property (ply, elem_name, &face_props[2]);
	  //ply_get_property (ply, elem_name, &face_props[3]);

      face_other = ply_get_other_properties (ply, elem_name,
                     offsetof(Face,other_props));

      /* grab all the face elements */
      for (j = 0; j < num_elems; j++) {
        //flist[j] = (Face *) malloc (sizeof (Face));
        ply_get_element (ply, (void *) &flist[j]);
      }
    }
    else
      other_elements = ply_get_other_element (ply, elem_name, num_elems);

	// free plist and its memory space in case of memory leaking
	if (plist)
	{
		for (int i = 0; i < nprops; i ++)
		{
			if (plist[i])
			{
				if (plist[i]->name)
				{
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

  //the ply variable is freed in the function - houtao
  ply_close (ply);
}
