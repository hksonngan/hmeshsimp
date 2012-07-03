/*

  Convert PLY models into SMF models.

  Michael Garland:  Oct 1996


  Based on a skeleton program
  (Greg Turk, August 1994)

*/

#include <stdmix.h>
#include "ply.h"
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
};

PlyProperty face_props[] = { /* list of property information for a vertex */
  {"vertex_indices", PLY_INT, PLY_INT, offsetof(Face,verts),
   1, PLY_UCHAR, PLY_UCHAR, offsetof(Face,nverts)},
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


  // modified by ht
  //ply  = ply_read (stdin, &nelems, &elist);
    //a space for PlyFile is malloced in the function
    //it must be freed after using, as well as elist - houtao
	ply = ply_open_for_reading(filename, &nelems, &elist, &file_type, &version);
	if(ply == NULL)
	{
		std::cout << "ply file open failed" << endl;
		return;
	}
 //def
 //PlyFile *ply_open_for_reading(
	// char *filename,
	// int *nelems,
	// char ***elem_names,
	// int *file_type,
	// float *version
	// )
  //ply_get_info (ply, &version, &file_type);

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
        ply_get_element (ply, (void *) &vlist[j]);
      }
    }
    else if (equal_strings ("face", elem_name)) {

      /* create a list to hold all the face elements */
      flist = (Face *) malloc (sizeof (Face) * num_elems);
      nfaces = num_elems;

      /* set up for getting face elements */

      ply_get_property (ply, elem_name, &face_props[0]);
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


/******************************************************************************
Write out the PLY file to standard out.
******************************************************************************/

void write_ply_model_to_file_as_smf(char *filename)
{
    int i, j;
    Vertex v;
    Face f;
	// modified by ht
	std::fstream fout(filename, fstream::out);

    fout << "# Generated from PLY data by ply2smf" << endl;
    fout << "# " << nverts << " vertices" << endl;
    fout << "# " << nfaces << " faces" << endl;

    for(i = 0; i < nverts; i ++)
    {
		v = vlist[i];
		fout << "v " << vlist[i].x << " " << vlist[i].y << " " << vlist[i].z << endl;
    }


    for(i=0; i<nfaces; i++)
    {
		f = flist[i];

		if( flist[i].nverts == 3 )
			fout << "f " << flist[i].verts[0]+1 << " "
			<< flist[i].verts[1]+1 << " "
			<< flist[i].verts[2]+1 << endl;
		else
		{
			fout << "f";
			for(j=0; j < flist[i].nverts; j++)
				fout << " " << flist[i].verts[j] + 1;
			fout << endl;
		}
    }

	fout.close();

	cerr << filename << " stored in disk" << endl;
}

/******************************************************************************
Main program.
******************************************************************************/

int ply2smf_entry(char *inputfile, char *outputfile)
{
	clean_ply();
	ply_read_file(inputfile);
	write_ply_model_to_file_as_smf(outputfile);
	clean_ply();

	//cerr << outputfile << " converted, stored locally in disk" << endl;

	return 0;
}
