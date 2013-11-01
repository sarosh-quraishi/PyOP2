# This file is part of PyOP2
#
# PyOP2 is Copyright (c) 2012, Imperial College London and
# others. Please see the AUTHORS file in the main source directory for
# a full list of copyright holders.  All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * The name of Imperial College London or that of other
#       contributors may not be used to endorse or promote products
#       derived from this software without specific prior written
#       permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTERS
# ''AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT HOLDERS OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
# INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
# STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
# OF THE POSSIBILITY OF SUCH DAMAGE.

from libcpp.vector cimport vector
from libcpp.set cimport set
from cython.operator cimport dereference as deref, preincrement as inc
from cpython cimport bool
import numpy as np
cimport numpy as np
import cython

np.import_array()

ctypedef np.int32_t DTYPE_t

ctypedef struct cmap:
    int from_size
    int from_exec_size
    int to_size
    int to_exec_size
    int arity
    int* values
    int* offset
    int layers

cdef cmap init_map(omap):
    cdef cmap out
    out.from_size = omap.iterset.size
    out.from_exec_size = omap.iterset.exec_size
    out.to_size = omap.toset.size
    out.to_exec_size = omap.toset.exec_size
    out.arity = omap.arity
    out.values = <int *>np.PyArray_DATA(omap.values_with_halo)
    out.offset = <int *>np.PyArray_DATA(omap.offset)
    out.layers = omap.iterset.layers
    return out


@cython.boundscheck(False)
@cython.wraparound(False)
cdef build_sparsity_pattern(int nrows, list maps):
    """Create and populate auxiliary data structure: for each element of the
    from set, for each row pointed to by the row map, add all columns pointed
    to by the col map."""
    cdef:
        int lsize, rsize, row, entry
        int e, i, d
        cmap rowmap, colmap
        vector[set[int]] s_diag, s_odiag

    lsize = nrows
    s_diag = vector[set[int]](lsize)
    s_odiag = vector[set[int]](lsize)

    for rmap, cmap in maps:
        rowmap = init_map(rmap)
        colmap = init_map(cmap)
        rsize = rowmap.from_exec_size;
        if rowmap.layers > 1:
            for e in range (rsize):
                for i in range(rowmap.arity):
                    for l in range(rowmap.layers - 1):
                        row = rowmap.values[i + e*rowmap.arity] + l * rowmap.offset[i]
                        # NOTE: this hides errors due to invalid map entries
                        if row < lsize:
                            for d in range(colmap.arity):
                                entry = colmap.values[d + e * colmap.arity] + l * colmap.offset[d]
                                if entry < lsize:
                                    s_diag[row].insert(entry)
                                else:
                                    s_odiag[row].insert(entry)
        else:
            for e in range (rsize):
                for i in range(rowmap.arity):
                    row = rowmap.values[i + e*rowmap.arity]
                    # NOTE: this hides errors due to invalid map entries
                    if row < lsize:
                        for d in range(colmap.arity):
                            entry = colmap.values[d + e * colmap.arity]
                            if entry < lsize:
                                s_diag[row].insert(entry)
                            else:
                                s_odiag[row].insert(entry)

    # Create final sparsity structure
    cdef np.ndarray[DTYPE_t, ndim=1] d_nnz = np.empty(lsize, dtype=np.int32)
    cdef np.ndarray[DTYPE_t, ndim=1] o_nnz = np.empty(lsize, dtype=np.int32)
    cdef np.ndarray[DTYPE_t, ndim=1] rowptr = np.empty(lsize + 1, dtype=np.int32)
    cdef int d_nz = 0
    cdef int o_nz = 0
    rowptr[0] = 0
    for row in range(lsize):
        d_nnz[row] = s_diag[row].size()
        d_nz += d_nnz[row]
        o_nnz[row] = s_odiag[row].size()
        rowptr[row+1] = rowptr[row] + d_nnz[row] + o_nnz[row]
        o_nz += o_nnz[row]

    cdef np.ndarray[DTYPE_t, ndim=1] colidx = np.empty(rowptr[lsize], dtype=np.int32)
    # Merge on and off diagonal pieces and sort.
    cdef np.ndarray[DTYPE_t, ndim=1] tmp = np.empty(d_nnz.max() + o_nnz.max(), dtype=np.int32)
    cdef int j, k, max_val = np.iinfo(tmp.dtype).max
    for row in range(lsize):
        tmp[:] = max_val        # Don't pick up bad values
        i = rowptr[row]
        j = 0
        it = s_diag[row].begin()
        while it != s_diag[row].end():
            tmp[j] = deref(it)
            inc(it)
            j += 1
        it = s_odiag[row].begin()
        while it != s_odiag[row].end():
            tmp[j] = deref(it)
            inc(it)
            j += 1
        tmp.sort()
        for k in range(j):
            colidx[i] = tmp[k]
            i += 1

    return d_nnz, o_nnz, d_nz, o_nz, rowptr, colidx

@cython.boundscheck(False)
@cython.wraparound(False)
def build_sparsity(object sparsity):
    cdef int nrows = sparsity._nrows
    cdef int lsize = nrows
    cdef int nmaps = len(sparsity._rmaps)

    sparsity._d_nnz, sparsity._o_nnz, sparsity._d_nz, sparsity._o_nz, sparsity._rowptr, sparsity._colidx = \
            build_sparsity_pattern(nrows, sparsity.maps)
