/* random number genertion, dragged from web */

#ifndef __U_RANDOM_USED_BY_HT__
#define __U_RANDOM_USED_BY_HT__

#ifdef __cplusplus
extern "C" {
#endif

void usrand (unsigned seed);
unsigned urand0 (void);
unsigned urand (void);

#define MAX_URAND 0xFFFFFFFFL

#ifdef __cplusplus
}
#endif

#endif