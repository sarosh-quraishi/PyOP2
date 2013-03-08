"""
Cython header file for LLVM vectorization library
"""

cdef extern from "llvm_vec.h":
    ctypedef struct pyop2_kernel_t:
        char* name
        char* src

    ctypedef enum pyop2_arg_type_t:
        PYOP2_DIRECT, PYOP2_INDIRECT, PYOP2_GLOBAL

    ctypedef enum pyop2_ctype_t:
        PYOP2_CHAR, PYOP2_UCHAR, PYOP2_SHORT, PYOP2_USHORT, PYOP2_INT, PYOP2_UINT, PYOP2_LONG, PYOP2_ULONG, PYOP2_LLONG, PYOP2_ULLONG, PYOP2_FLOAT, PYOP2_DOUBLE

    ctypedef enum pyop2_access_t:
        PYOP2_READ, PYOP2_WRITE, PYOP2_RW, PYOP2_INC, PYOP2_MIN, PYOP2_MAX

    ctypedef struct pyop2_arg_t:
        pyop2_arg_type_t type
        pyop2_ctype_t dat_ctype
        int dat_dim
        int map_dim
        int index
        pyop2_access_t access

    void llvm_vectorize(pyop2_kernel_t*, pyop2_arg_t*)