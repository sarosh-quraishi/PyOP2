#ifndef LLVM_VEC_H
#define LLVM_VEC_H

typedef struct {
    char* name;
    char* src;
} pyop2_kernel_t;

typedef enum {
    PYOP2_DIRECT,
    PYOP2_INDIRECT,
    PYOP2_GLOBAL,
} pyop2_arg_type_t;

typedef enum {
    PYOP2_CHAR,
    PYOP2_UCHAR,
    PYOP2_SHORT,
    PYOP2_USHORT,
    PYOP2_INT,
    PYOP2_UINT,
    PYOP2_LONG,
    PYOP2_ULONG,
    PYOP2_LLONG,
    PYOP2_ULLONG,
    PYOP2_FLOAT,
    PYOP2_DOUBLE,
} pyop2_ctype_t;

typedef enum {
    PYOP2_READ,
    PYOP2_WRITE,
    PYOP2_RW,
    PYOP2_INC,
    PYOP2_MIN,
    PYOP2_MAX,
} pyop2_access_t;

typedef struct {
    pyop2_arg_type_t type;

    pyop2_ctype_t dat_ctype;
    int dat_dim;
    
    int map_dim;           // irrelevant unless type == PYOP2_INDIRECT

    int index;             // irrelevant unless type == PYOP2_INDIRECT

    pyop2_access_t access; // valid values dependand on 'type'
                           //  - PYOP2_DIRECT: READ, WRITE, RW
                           //  - PYOP2_INDIRECT: all valid
                           //  - PYOP2_GLOBAL: READ, INC, MIN, MAX
} pyop2_arg_t;

void
llvm_vectorize(pyop2_kernel_t*, int, pyop2_arg_t*);

#endif /* LLVM_VEC_H */