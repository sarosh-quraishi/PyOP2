Lazy Evaluation Documentation
=============================

Approach
--------

Unless explicitly set up to execute greedily, parloops are not executed immediatly. In doing so, it opens PyOP2 to a range of execution rescheduling and program transformations at runtime.

Delaying computations
---------------------

In order to delay the execution of parloops, ParLoop objects inherit from `base::LazyComputation` and must implement the execution of the parloop in the `_run` method. LazyComputation objects implements an `enqueue` method that stores the computation to be delayed into a list structure, `base::ExecutionTrace`, for later execution.

Declaring computations
----------------------

For the purpose of reordering the execution of parloops while preserving the correctness of the PyOP2 program, when inheriting from `LazyComputation`, `ParLoop` objects must declare their input/output dependencies, that is the set of PyOP2 Data Objects that read and/or written during the execution of the parloop. This is directly derived from the arguments of the `ParLoop` (see instance attributes `reads` and `writes` of class `LazyComputation`).

Forcing computations
--------------------

In turns to access the up-to-date content of PyOP2 Data objects, these must force the execution of delayed parloops before it can be accessed. This is done by calling the `evaluate` method of the `ExecutionTrace` object from the PyOP2 Data objects public accessors, passing the reads and writes dependencies to be updated, (eg, `evaluate(reads={a,b}, writes={b}`, tells `a` and `b` should be updated with the intent of being read, and read and writen respectively).

Propagating dependencies
------------------------

Method `evaluate` determine which of the computation which have been delayed that now must be execute in order to update the reads and writes dependencies passed as arguments. This method iterate the list of delayed computation in reverse order: from the most to the least recently delayed computation.

Let us call *Ra*, *Wa*, *Rc* and *Wc*, the read (R) and write (W) dependencies passed as arguments (a) and of the delayed computation (c) respectively. Computation `c`, is required to be executed if:
    Ra . Wc + Wa . Rc + Wa . Wc
is not empty: *if a dependency read is writen by c, if a depency writen must be read first, or if a dependency is being overwriten (preserving write ordering)*

if `c` is required for *Ra* or *Wa* then *Ra* and *Wa* become:
    Ra = Ra + Rc - Wc
    Wa = Wa + Wc
*New write dependencies need to be propagated, but read dependencies that will be updated need not.*

Once the iteration of the list is over, dependant computations are executed in oldest to most recent order and removed from the list.

Notes:
------

lazy-split branch
~~~~~~~~~~~~~~~~~

Changes:

* `ParLoop` no longer inherits from `LazyComputation`, instead, ParLoop constructor instanciate `LazyComputation` object (start halo exchange, compute core elements, finish halo exchange, compute owned elements, compute halo elements). (To avoid a circular dependency problem in the code)

* Helper class (CORE, OWNED, HALO) help create finer dependencies for PyOP2 Data objects.

* instead of enqueuing halo exchange computation at the end of the trace, they are push as far back as possible (as long as the previous computation is independant of the halo echange)


Optimising dependencies propagation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

optimise `evaluate`: should merge test for dependant computations with dependencies propagation

