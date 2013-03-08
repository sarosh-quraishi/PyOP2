from libc.stdlib cimport malloc, free
cimport _llvm_vec as llvm

def llvm_vectorize(kernel, *args):
    cdef llvm.pyop2_kernel_t k
    cdef llvm.pyop2_arg_t* a

    k.name = kernel.name
    k.src = kernel.src

    def arg_type(arg):
        if arg.is_direct:
            return PYOP2_DIRECT
        elif arg.is_global:
            return PYOP2_GLOBAL
        else:
            return PYOP2_INDIRECT

    def dat_ctype(arg):
        conv = { "bool": PYOP2_UCHAR,
                 "int8": PYOP2_CHAR,
                 "int16": PYOP2_SHORT,
                 "int32": PYOP2_INT,
                 "int64": PYOP2_LLONG,
                 "uint8": PYOP2_UCHAR,
                 "uint16": PYOP2_USHORT,
                 "uint32": PYOP2_UINT,
                 "uint64": PYOP2_ULLONG,
                 "float": PYOP2_DOUBLE,
                 "float32": PYOP2_FLOAT,
                     "float64": PYOP2_DOUBLE }
        return conv[arg.data.dtype.name]

    def arg_access(arg):
        conv = { OP_READ: PYOP2_READ,
                 OP_WRITE: PYOP2_WRITE,
                 OP_RW: PYOP2_RW,
                 OP_INC: PYOP2_INC,
                 OP_MIN: PYOP2_MIN,
                 OP_MAX: PYOP2_MAX }
        return conv[arg.access]

    a = <pyop2_arg_t*> malloc(sizeof(pyop2_arg_t) * len(*args))
    for i, arg in enumerate(*args):
        a[i].type = arg_type(arg)
        a[i].dat_ctype = dat_ctype(arg)
        a[i].dat_dim = arg.data.cdim
        a[i].map_dim = arg.map.dim
        a[i].index = arg.index
        a[i].access = arg_access(arg)

    llvm.llvm_vectorize(k, a)
    free(a)
