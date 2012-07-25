/*************************************
 author   : houtao
 spec     : include file used by the 
			HMesh project
 email    : waytofall916@gmail.com
 copyleft : basically GPL
*************************************/

/* caution!! this file as well as 'ply_additional.h' may be abandoned due to the 'PlyStream' class defined */

#ifndef __PLY_INC_H__
#define __PLY_INC_H__

//#ifdef __cplusplus
//extern "C" {
//#endif

#include "ply.h"

typedef struct Vertex {
	float	x, y, z;
	float	r, g, b;
	void	*other_props;       /* other properties */
} Vertex;

typedef struct Face {
	unsigned char	nverts;    /* number of vertex indices in list */
	int		*verts;              /* vertex index list */
	float	r, g, b;
	void	*other_props;       /* other properties */
} Face;

extern PlyProperty vert_props[];
extern PlyProperty face_props[];


/*** the PLY object ***/

extern int nverts, nfaces;
//extern Vertex **vlist;
//extern Face **flist;
extern Vertex *vlist;
extern Face *flist;
extern int nelems;
extern char **elist;
extern PlyOtherElems *other_elements;
extern PlyOtherProp *vert_other, *face_other;
extern int num_comments;
extern char **comments;
extern int num_obj_info;
extern char **obj_info;
extern int file_type;

// added by houtao
extern int ply2smf_entry(char *inputfile, char *outputfile);
extern void ply_read_file(char* filename);
extern void write_ply_model_to_file_as_smf(char *filename);
extern void clean_ply();
//extern int convert_ply(const char* infilename, const char* outfilename, int verbose_flag);

//#ifdef __cplusplus
//}
//#endif

#endif /* !__PLY_INC_H__ */