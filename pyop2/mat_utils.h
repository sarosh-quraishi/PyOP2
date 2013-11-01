#ifndef _MAT_UTILS_H
#define _MAT_UTILS_H

#include <petscmat.h>

void addto(Mat mat, const void *values,
           int nrows, int *rows,
           int ncols, int *cols,
           int insert);

#endif // _MAT_UTILS_H
