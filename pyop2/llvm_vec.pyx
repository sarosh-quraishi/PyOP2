from pyop2.runtime_base import READ, WRITE, RW, INC, MIN, MAX
from libc.stdlib cimport malloc, free
cimport _llvm_vec as llvm

def llvm_vectorize(kernel, *args):
    cdef llvm.pyop2_kernel_t k
    cdef llvm.pyop2_arg_t* a

    k.name = kernel.name
    k.src = kernel.code

    def arg_type(arg):
        if arg._is_direct:
            return llvm.PYOP2_DIRECT
        elif arg._is_global:
            return llvm.PYOP2_GLOBAL
        else:
            return llvm.PYOP2_INDIRECT

    def dat_ctype(arg):
        conv = { "bool": llvm.PYOP2_UCHAR,
                 "int8": llvm.PYOP2_CHAR,
                 "int16": llvm.PYOP2_SHORT,
                 "int32": llvm.PYOP2_INT,
                 "int64": llvm.PYOP2_LLONG,
                 "uint8": llvm.PYOP2_UCHAR,
                 "uint16": llvm.PYOP2_USHORT,
                 "uint32": llvm.PYOP2_UINT,
                 "uint64": llvm.PYOP2_ULLONG,
                 "float": llvm.PYOP2_DOUBLE,
                 "float32": llvm.PYOP2_FLOAT,
                     "float64": llvm.PYOP2_DOUBLE }
        return conv[arg.data.dtype.name]

    def arg_access(arg):
        conv = { READ: llvm.PYOP2_READ,
                 WRITE: llvm.PYOP2_WRITE,
                 RW: llvm.PYOP2_RW,
                 INC: llvm.PYOP2_INC,
                 MIN: llvm.PYOP2_MIN,
                 MAX: llvm.PYOP2_MAX }
        return conv[arg.access]

    a = <llvm.pyop2_arg_t*> malloc(sizeof(llvm.pyop2_arg_t) * len(args))
    for i, arg in enumerate(args):
        a[i].type = arg_type(arg)
        a[i].dat_ctype = dat_ctype(arg)
        a[i].dat_dim = arg.data.cdim
        a[i].map_dim = arg.map.dim
        a[i].index = arg.index
        a[i].access = arg_access(arg)

    llvm.llvm_vectorize(&k, <int> len(args), a)
    free(a)
