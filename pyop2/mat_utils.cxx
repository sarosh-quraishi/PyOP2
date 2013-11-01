#include <assert.h>
#include "mat_utils.h"

void addto(Mat mat, const void *values,
           int nrows, int *rows,
           int ncols, int *cols,
           int insert)
{
  PetscInt rbs, cbs;
  int i, j;
  int all_zero = 1;
  // FIMXE: this assumes we're getting a PetscScalar
  const PetscScalar *v = (const PetscScalar *)values;
  assert(mat && value);

  MatSetValuesBlockedLocal( mat,
                            nrows, (const PetscInt *)rows,
                            ncols, (const PetscInt *)cols,
                            v, insert ? INSERT_VALUES : ADD_VALUES );
}
