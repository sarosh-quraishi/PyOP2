.. _backends:

PyOP2 Backends
==============

PyOP2 supports a number of different backends to be able to run parallel
computations on different hardware architectures. The currently supported
backends are

* ``sequential``: runs sequentially on a single CPU core.
* ``openmp``: runs multiple threads on an SMP CPU using OpenMP. The number of
  threads is set with the environment variable ``OMP_NUM_THREADS``.
* ``cuda``: offloads computation to a NVIDA GPU (requires :ref:`CUDA and pycuda
  <cuda-installation>`)
* ``opencl``: offloads computation to an OpenCL device, either a multi-core
  CPU or a GPU (requires :ref:`OpenCL and pyopencl <opencl-installation>`)

The ``sequential`` and ``openmp`` backends also support distributed parallel
computations using MPI. For OpenMP this means a hybrid parallel execution
with ``OMP_NUM_THREADS`` threads per MPI rank. Datastructures must be suitably
partitioned in this case with overlapping regions, so called halos. These are
described in detail in :doc:`mpi`.

Sequential backend
------------------

Any computation in PyOP2 requires generating code at runtime specific to each
individual :func:`~pyop2.par_loop`. The sequential backend generates code via
the `Instant`_ utility from the `FEniCS project`_. Since there is no parallel
computation for the sequential backend, the generated code is a C wrapper
function with a ``for`` loop calling the kernel for the respective
:func:`~pyop2.par_loop`. This wrapper also takes care of staging in and out
the data as requested by the access descriptors requested in the parallel
loop. Both the kernel and the wrapper function are just-in-time compiled in a
single compilation unit such that the kernel call can be inlined and does not
incur any function call overhead.

Recall the :func:`~pyop2.par_loop` calling the ``midpoint`` kernel from
:doc:`kernels`: ::

  op2.par_loop(midpoint, cells,
               midpoints(op2.WRITE),
               coordinates(op2.READ, cell2vertex))
 
.. highlight:: c
   :linenothreshold: 5

The JIT compiled code for this loop is the kernel followed by the generated
wrapper code: ::

  inline void midpoint(double p[2], double *coords[2]) {
    p[0] = (coords[0][0] + coords[1][0] + coords[2][0]) / 3.0;
    p[1] = (coords[0][1] + coords[1][1] + coords[2][1]) / 3.0;
  }

  void wrap_midpoint__(PyObject *_start, PyObject *_end,
                       PyObject *_arg0_0,
                       PyObject *_arg1_0, PyObject *_arg1_0_map0_0) {
    int start = (int)PyInt_AsLong(_start);
    int end = (int)PyInt_AsLong(_end);
    double *arg0_0 = (double *)(((PyArrayObject *)_arg0_0)->data);
    double *arg1_0 = (double *)(((PyArrayObject *)_arg1_0)->data);
    int *arg1_0_map0_0 = (int *)(((PyArrayObject *)_arg1_0_map0_0)->data);
    double *arg1_0_vec[3];
    for ( int n = start; n < end; n++ ) {
      int i = n;
      arg1_0_vec[0] = arg1_0 + arg1_0_map0_0[i * 3 + 0] * 2;
      arg1_0_vec[1] = arg1_0 + arg1_0_map0_0[i * 3 + 1] * 2;
      arg1_0_vec[2] = arg1_0 + arg1_0_map0_0[i * 3 + 2] * 2;
      midpoint(arg0_0 + i * 2, arg1_0_vec);
    }
  }

Note that the wrapper function is called directly from Python and therefore
all arguments are plain Python objects, which first need to be unwrapped. The
arguments ``_start`` and ``_end`` define the iteration set indices to iterate
over. The remaining arguments are :class:`arrays <numpy.ndarray>`
corresponding to a :class:`~pyop2.Dat` or :class:`~pyop2.Map` passed to the
:func:`~pyop2.par_loop`. Arguments are consecutively numbered to avoid name
clashes.

The first :func:`~pyop2.par_loop` argument ``midpoints`` is direct and
therefore no corresponding :class:`~pyop2.Map` is passed to the wrapper
function and the data pointer is passed straight to the kernel with an
appropriate offset. The second arguments ``coordinates`` is indirect and hence
a :class:`~pyop2.Dat`-:class:`~pyop2.Map` pair is passed. Pointers to the data
are gathered via the :class:`~pyop2.Map` of arity 3 and staged in the array
``arg1_0_vec``, which is passed to kernel. The coordinate data can therefore
be accessed in the kernel via double indirection as if it was stored
consecutively in memory. Note that for both arguments, the pointers are to two
consecutive double values, since the :class:`~pyop2.DataSet` is of dimension
two in either case.

OpenMP backend
--------------

The OpenMP uses the same infrastructure for code generation and JIT
compilation as the sequential backend described above. In contrast however,
the ``for`` loop is annotated with OpenMP pragmas to make it execute in
parallel with multiple threads. To avoid race conditions on data access, the
iteration set is coloured and a thread safe execution plan is computed as
described in :doc:`colouring`.

The JIT compiled code for the parallel loop from above changes as follows: ::

  void wrap_midpoint__(PyObject* _boffset,
                       PyObject* _nblocks,
                       PyObject* _blkmap,
                       PyObject* _offset,
                       PyObject* _nelems,
                       PyObject *_arg0_0,
                       PyObject *_arg1_0, PyObject *_arg1_0_map0_0) {
    int boffset = (int)PyInt_AsLong(_boffset);
    int nblocks = (int)PyInt_AsLong(_nblocks);
    int* blkmap = (int *)(((PyArrayObject *)_blkmap)->data);
    int* offset = (int *)(((PyArrayObject *)_offset)->data);
    int* nelems = (int *)(((PyArrayObject *)_nelems)->data);
    double *arg0_0 = (double *)(((PyArrayObject *)_arg0_0)->data);
    double *arg1_0 = (double *)(((PyArrayObject *)_arg1_0)->data);
    int *arg1_0_map0_0 = (int *)(((PyArrayObject *)_arg1_0_map0_0)->data);
    double *arg1_0_vec[32][3];
    #ifdef _OPENMP
    int nthread = omp_get_max_threads();
    #else
    int nthread = 1;
    #endif
    #pragma omp parallel shared(boffset, nblocks, nelems, blkmap)
    {
      int tid = omp_get_thread_num();
      #pragma omp for schedule(static)
      for (int __b = boffset; __b < boffset + nblocks; __b++)
      {
        int bid = blkmap[__b];
        int nelem = nelems[bid];
        int efirst = offset[bid];
        for (int n = efirst; n < efirst+ nelem; n++ )
        {
          int i = n;
          arg1_0_vec[tid][0] = arg1_0 + arg1_0_map0_0[i * 3 + 0] * 2;
          arg1_0_vec[tid][1] = arg1_0 + arg1_0_map0_0[i * 3 + 1] * 2;
          arg1_0_vec[tid][2] = arg1_0 + arg1_0_map0_0[i * 3 + 2] * 2;
          midpoint(arg0_0 + i * 2, arg1_0_vec[tid]);
        }
      }
    }
  }

Computation is split in ``nblocks`` blocks which start at an initial offset
``boffset`` and correspond to colours that can be executed conflict free in
parallel. This loop over colours is therefore wrapped in an OpenMP parallel
region and is annotated with an ``omp for`` pragma. The block id ``bid`` for
each of these blocks is given by the block map ``blkmap`` and is the index
into the arrays ``nelems`` and ``offset`` provided as part of the execution
plan. These are the number of elements that are part of the given block and
its starting index. Note that each thread needs its own staging array
``arg1_0_vec``, which is therefore scoped by the thread id.

CUDA backend
------------

The CUDA backend makes extensive use of PyCUDA_ and its infrastructure for
just-in-time compilation of CUDA kernels. Linear solvers and sparse matrix
data structures are implemented on top of the `CUSP library`_ and are
described in greater detail in :doc:`linear_algebra`. Code generation uses a
template based approach, where a ``__global__`` stub routine to be called from
the host is generated, which takes care of data marshaling and calling the
user kernel as an inline ``__device__`` function.

When the :func:`~pyop2.par_loop` is called, PyOP2 uses the access descriptors
to determine which data needs to be transfered from host host to device prior
to launching the kernel and which data needs to brought back to the host
afterwards. All data transfer is triggered lazily i.e. the actual copy only
occurs once the data is requested. Flags indicate the state of a given
:class:`~pyop2.Dat` at any point in time:

* ``DEVICE_UNALLOCATED``: no data is allocated on the device
* ``HOST_UNALLOCATED``: no data is allocated on the host
* ``DEVICE``: data is up-to-date (valid) on the device, but invalid on the
  host
* ``HOST``: data is up-to-date (valid) on the host, but invalid on the device
* ``BOTH``: data is up-to-date (valid) on both the host and device

We consider the same ``midpoint`` kernel as in the previous examples, which
requires no modification and is automatically annonated with a ``__device__``
qualifier. PyCUDA_ takes care of generating a host stub for the generated
kernel stub ``__midpoint_stub`` given a list of parameter types. It takes care
of translating Python objects to plain C data types and pointers, such that a
CUDA kernel can be launched straight from Python. The entire CUDA code PyOP2
generates is as follows: ::

  __device__ void midpoint(double p[2], double *coords[2])
  {
    p[0] = ((coords[0][0] + coords[1][0]) + coords[2][0]) / 3.0;
    p[1] = ((coords[0][1] + coords[1][1]) + coords[2][1]) / 3.0;
  }

  __global__ void __midpoint_stub(int set_size, int set_offset,
      double *arg0,
      double *ind_arg1,
      int *ind_map,
      short *loc_map,
      int *ind_sizes,
      int *ind_offs,
      int block_offset,
      int *blkmap,
      int *offset,
      int *nelems,
      int *nthrcol,
      int *thrcol,
      int nblocks) {
    extern __shared__ char shared[];
    __shared__ int *ind_arg1_map;
    __shared__ int ind_arg1_size;
    __shared__ double * ind_arg1_shared;
    __shared__ int nelem, offset_b, offset_b_abs;
    
    double *ind_arg1_vec[3];

    if (blockIdx.x + blockIdx.y * gridDim.x >= nblocks) return;
    if (threadIdx.x == 0) {
      int blockId = blkmap[blockIdx.x + blockIdx.y * gridDim.x + block_offset];
      nelem = nelems[blockId];
      offset_b_abs = offset[blockId];
      offset_b = offset_b_abs - set_offset;

      ind_arg1_size = ind_sizes[0 + blockId * 1];
      ind_arg1_map = &ind_map[0 * set_size] + ind_offs[0 + blockId * 1];
      
      int nbytes = 0;
      ind_arg1_shared = (double *) &shared[nbytes];
    }

    __syncthreads();

    // Copy into shared memory
    for ( int idx = threadIdx.x; idx < ind_arg1_size * 2; idx += blockDim.x ) {
      ind_arg1_shared[idx] = ind_arg1[idx % 2 + ind_arg1_map[idx / 2] * 2];
    }
    
    __syncthreads();

    // process set elements
    for ( int idx = threadIdx.x; idx < nelem; idx += blockDim.x ) {
      ind_arg1_vec[0] = ind_arg1_shared + loc_map[0*set_size + idx + offset_b]*2;
      ind_arg1_vec[1] = ind_arg1_shared + loc_map[1*set_size + idx + offset_b]*2;
      ind_arg1_vec[2] = ind_arg1_shared + loc_map[2*set_size + idx + offset_b]*2;

      midpoint(arg0 + 2 * (idx + offset_b_abs), ind_arg1_vec);
    }
  }

The CUDA kernel ``__midpoint_stub`` is launched on the GPU for a specific
number of threads. Each thread is identified inside the kernel by its thread
id ``threadIdx`` within a block of threads identified by a two dimensional
block id ``blockIdx`` within a grid of blocks.

As for OpenMP, there is the potential for data races, which are prevented by
colouring the iteration set and computing a parallel execution plan, where all
elements of the same colour can be modified simultaneously. Each colour is
computed by a block of threads in parallel. All threads of a thread block have
access to a shared memory, which is used as a shared staging area initialised
by thread 0 of each block, see lines 30-41 above. A call to
``__syncthreads()`` makes sure these initial values are visible to all threads
of the block. Afterwards, all threads cooperatively gather data from the
indirectly accessed :class:`~pyop2.Dat` via the :class:`~pyop2.Map`, followed
by another synchronisation. Following that, each thread stages pointers to
coordinate data in a thread-private array which is then passed to the
``midpoint`` kernel. As for other backends, the first argument, which is
written directly, is passed as a pointer to global device memory with a
suitable offset.

.. _Instant: https://bitbucket.org/fenics-project/instant
.. _FEniCS project: http://fenicsproject.org
.. _PyCUDA: http://mathema.tician.de/software/pycuda/
.. _CUSP library: http://cusplibrary.github.io
