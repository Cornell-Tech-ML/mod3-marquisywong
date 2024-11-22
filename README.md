# MiniTorch Module 3

<img src="https://minitorch.github.io/minitorch.svg" width="50%">

* Docs: https://minitorch.github.io/

* Overview: https://minitorch.github.io/module3.html


You will need to modify `tensor_functions.py` slightly in this assignment.

* Tests:

```
python run_tests.py
```

* Note:

Several of the tests for this assignment will only run if you are on a GPU machine and will not
run on github's test infrastructure. Please follow the instructions to setup up a colab machine
to run these tests.

This assignment requires the following files from the previous assignments. You can get these by running

```bash
python sync_previous_module.py previous-module-dir current-module-dir
```

The files that will be synced are:

        minitorch/tensor_data.py minitorch/tensor_functions.py minitorch/tensor_ops.py minitorch/operators.py minitorch/scalar.py minitorch/scalar_functions.py minitorch/module.py minitorch/autodiff.py minitorch/module.py project/run_manual.py project/run_scalar.py project/run_tensor.py minitorch/operators.py minitorch/module.py minitorch/autodiff.py minitorch/tensor.py minitorch/datasets.py minitorch/testing.py minitorch/optim.py


Task 3.1 and 3.2, output from:
python project/parallel_check.py 

(.venv) marquiswong@whale mod3-marquisywong % python project/parallel_check.py
MAP
 
================================================================================
 Parallel Accelerator Optimizing:  Function tensor_map.<locals>._map, 
/Users/marquiswong/workspace/mod3-marquisywong/minitorch/fast_ops.py (163)  
================================================================================


Parallel loop listing for  Function tensor_map.<locals>._map, /Users/marquiswong/workspace/mod3-marquisywong/minitorch/fast_ops.py (163) 
-----------------------------------------------------------------------------|loop #ID
    def _map(                                                                | 
        out: Storage,                                                        | 
        out_shape: Shape,                                                    | 
        out_strides: Strides,                                                | 
        in_storage: Storage,                                                 | 
        in_shape: Shape,                                                     | 
        in_strides: Strides,                                                 | 
    ) -> None:                                                               | 
        if np.array_equal(in_strides, out_strides) and np.array_equal(       | 
            in_shape, out_shape                                              | 
        ):                                                                   | 
            for x in prange(-------------------------------------------------| #0
                len(out)                                                     | 
            ):  #parallel loop                                               | 
                out[x] = fn(                                                 | 
                    in_storage[x]                                            | 
                )  #fn to input elem                                         | 
                                                                             | 
        else:                                                                | 
            for x in prange(-------------------------------------------------| #1
                len(out)                                                     | 
            ):  #parallel loop                                               | 
                out_index: Index = np.empty(MAX_DIMS, dtype=np.int32)        | 
                in_index: Index = np.empty(MAX_DIMS, dtype=np.int32)         | 
                                                                             | 
                #multi-dim index                                             | 
                to_index(x, out_shape, out_index)                            | 
                                                                             | 
                #corresponding input index                                   | 
                broadcast_index(out_index, out_shape, in_shape, in_index)    | 
                o = index_to_position(out_index, out_strides)  #output       | 
                i = index_to_position(in_index, in_strides)  #input          | 
                                                                             | 
                out[o] = fn(in_storage[i])                                   | 
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 2 parallel for-
loop(s) (originating from loops labelled: #0, #1).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
 
---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at 
/Users/marquiswong/workspace/mod3-marquisywong/minitorch/fast_ops.py (185) is 
hoisted out of the parallel loop labelled #1 (it will be performed before the 
loop is executed and reused inside the loop):
   Allocation:: out_index: Index = np.empty(MAX_DIMS, dtype=np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at 
/Users/marquiswong/workspace/mod3-marquisywong/minitorch/fast_ops.py (186) is 
hoisted out of the parallel loop labelled #1 (it will be performed before the 
loop is executed and reused inside the loop):
   Allocation:: in_index: Index = np.empty(MAX_DIMS, dtype=np.int32)
    - numpy.empty() is used for the allocation.
None
ZIP
 
================================================================================
 Parallel Accelerator Optimizing:  Function tensor_zip.<locals>._zip, 
/Users/marquiswong/workspace/mod3-marquisywong/minitorch/fast_ops.py (225)  
================================================================================


Parallel loop listing for  Function tensor_zip.<locals>._zip, /Users/marquiswong/workspace/mod3-marquisywong/minitorch/fast_ops.py (225) 
---------------------------------------------------------------------------|loop #ID
    def _zip(                                                              | 
        out: Storage,                                                      | 
        out_shape: Shape,                                                  | 
        out_strides: Strides,                                              | 
        a_storage: Storage,                                                | 
        a_shape: Shape,                                                    | 
        a_strides: Strides,                                                | 
        b_storage: Storage,                                                | 
        b_shape: Shape,                                                    | 
        b_strides: Strides,                                                | 
    ) -> None:                                                             | 
        if (                                                               | 
            np.array_equal(a_strides, b_strides)                           | 
            and np.array_equal(a_strides, out_strides)                     | 
            and np.array_equal(a_shape, b_shape)                           | 
            and np.array_equal(a_shape, out_shape)                         | 
        ):                                                                 | 
            for i in prange(len(out)):  #parallel loop---------------------| #2
                out[i] = fn(a_storage[i], b_storage[i])                    | 
        else:                                                              | 
            for i in prange(len(out)):  #parallel loop---------------------| #3
                                                                           | 
                out_index: Index = np.empty(MAX_DIMS, dtype=np.int32)      | 
                a_index: Index = np.empty(MAX_DIMS, dtype=np.int32)        | 
                b_index: Index = np.empty(MAX_DIMS, dtype=np.int32)        | 
                                                                           | 
                #multi-dim index, handle corresponding indices             | 
                to_index(i, out_shape, out_index)                          | 
                broadcast_index(out_index, out_shape, a_shape, a_index)    | 
                broadcast_index(out_index, out_shape, b_shape, b_index)    | 
                                                                           | 
                                                                           | 
                #flat storage positions                                    | 
                o = index_to_position(out_index, out_strides)              | 
                a = index_to_position(a_index, a_strides)                  | 
                b = index_to_position(b_index, b_strides)                  | 
                                                                           | 
                                                                           | 
                out[o] = fn(a_storage[a], b_storage[b])                    | 
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 2 parallel for-
loop(s) (originating from loops labelled: #2, #3).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
 
---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at 
/Users/marquiswong/workspace/mod3-marquisywong/minitorch/fast_ops.py (247) is 
hoisted out of the parallel loop labelled #3 (it will be performed before the 
loop is executed and reused inside the loop):
   Allocation:: out_index: Index = np.empty(MAX_DIMS, dtype=np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at 
/Users/marquiswong/workspace/mod3-marquisywong/minitorch/fast_ops.py (248) is 
hoisted out of the parallel loop labelled #3 (it will be performed before the 
loop is executed and reused inside the loop):
   Allocation:: a_index: Index = np.empty(MAX_DIMS, dtype=np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at 
/Users/marquiswong/workspace/mod3-marquisywong/minitorch/fast_ops.py (249) is 
hoisted out of the parallel loop labelled #3 (it will be performed before the 
loop is executed and reused inside the loop):
   Allocation:: b_index: Index = np.empty(MAX_DIMS, dtype=np.int32)
    - numpy.empty() is used for the allocation.
None
REDUCE
 
================================================================================
 Parallel Accelerator Optimizing:  Function tensor_reduce.<locals>._reduce, 
/Users/marquiswong/workspace/mod3-marquisywong/minitorch/fast_ops.py (289)  
================================================================================


Parallel loop listing for  Function tensor_reduce.<locals>._reduce, /Users/marquiswong/workspace/mod3-marquisywong/minitorch/fast_ops.py (289) 
---------------------------------------------------------------------|loop #ID
    def _reduce(                                                     | 
        out: Storage,                                                | 
        out_shape: Shape,                                            | 
        out_strides: Strides,                                        | 
        a_storage: Storage,                                          | 
        a_shape: Shape,                                              | 
        a_strides: Strides,                                          | 
        reduce_dim: int,                                             | 
    ) -> None:                                                       | 
        for x in prange(len(out)):-----------------------------------| #4
            out_index: Index = np.empty(MAX_DIMS, dtype=np.int32)    | 
            #multi-dim index                                         | 
            to_index(x, out_shape, out_index)                        | 
            o = index_to_position(out_index, out_strides) #output    | 
            i = index_to_position(out_index, a_strides) #input       | 
                                                                     | 
            #accumulator using output tensor                         | 
            acc = out[o]                                             | 
                                                                     | 
            #handle strides for reduction                            | 
            step = a_strides[reduce_dim]                             | 
            for _ in range(a_shape[reduce_dim]):                     | 
                acc = fn(acc, a_storage[i])                          | 
                i += step                                            | 
                                                                     | 
            out[o] = acc                                             | 
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 1 parallel for-
loop(s) (originating from loops labelled: #4).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
 
---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at 
/Users/marquiswong/workspace/mod3-marquisywong/minitorch/fast_ops.py (299) is 
hoisted out of the parallel loop labelled #4 (it will be performed before the 
loop is executed and reused inside the loop):
   Allocation:: out_index: Index = np.empty(MAX_DIMS, dtype=np.int32)
    - numpy.empty() is used for the allocation.
None
MATRIX MULTIPLY
 
================================================================================
 Parallel Accelerator Optimizing:  Function _tensor_matrix_multiply, 
/Users/marquiswong/workspace/mod3-marquisywong/minitorch/fast_ops.py (320)  
================================================================================


Parallel loop listing for  Function _tensor_matrix_multiply, /Users/marquiswong/workspace/mod3-marquisywong/minitorch/fast_ops.py (320) 
------------------------------------------------------------------------------------------------|loop #ID
def _tensor_matrix_multiply(                                                                    | 
    out: Storage,                                                                               | 
    out_shape: Shape,                                                                           | 
    out_strides: Strides,                                                                       | 
    a_storage: Storage,                                                                         | 
    a_shape: Shape,                                                                             | 
    a_strides: Strides,                                                                         | 
    b_storage: Storage,                                                                         | 
    b_shape: Shape,                                                                             | 
    b_strides: Strides,                                                                         | 
) -> None:                                                                                      | 
    """NUMBA tensor matrix multiply function.                                                   | 
                                                                                                | 
    Should work for any tensor shapes that broadcast as long as                                 | 
                                                                                                | 
    ```                                                                                         | 
    assert a_shape[-1] == b_shape[-2]                                                           | 
    ```                                                                                         | 
                                                                                                | 
    Optimizations:                                                                              | 
                                                                                                | 
    * Outer loop in parallel                                                                    | 
    * No index buffers or function calls                                                        | 
    * Inner loop should have no global writes, 1 multiply.                                      | 
                                                                                                | 
                                                                                                | 
    Args:                                                                                       | 
    ----                                                                                        | 
        out (Storage): storage for `out` tensor                                                 | 
        out_shape (Shape): shape for `out` tensor                                               | 
        out_strides (Strides): strides for `out` tensor                                         | 
        a_storage (Storage): storage for `a` tensor                                             | 
        a_shape (Shape): shape for `a` tensor                                                   | 
        a_strides (Strides): strides for `a` tensor                                             | 
        b_storage (Storage): storage for `b` tensor                                             | 
        b_shape (Shape): shape for `b` tensor                                                   | 
        b_strides (Strides): strides for `b` tensor                                             | 
                                                                                                | 
    Returns:                                                                                    | 
    -------                                                                                     | 
        None : Fills in `out`                                                                   | 
                                                                                                | 
    """                                                                                         | 
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0                                      | 
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0                                      | 
                                                                                                | 
    assert a_shape[-1] == b_shape[-2]                                                           | 
                                                                                                | 
                                                                                                | 
    for batch in prange(out_shape[0]):----------------------------------------------------------| #5
        #rows and cols                                                                          | 
        for cols in range(out_shape[-1]):  #column                                              | 
            for rows in range(out_shape[-2]):  #row                                             | 
                                                                                                | 
                #starting positions                                                             | 
                a_pos = batch * a_batch_stride + rows * a_strides[-2]                           | 
                b_pos = batch * b_batch_stride + cols * b_strides[-1]                           | 
                                                                                                | 
                acc = 0.0                                                                       | 
                                                                                                | 
                #reduction loop                                                                 | 
                for _ in range(a_shape[-1]):                                                    | 
                    acc += (                                                                    | 
                        a_storage[a_pos] * b_storage[b_pos]                                     | 
                    )  #accum dot prod                                                          | 
                    a_pos += a_strides[-1]                                                      | 
                    b_pos += b_strides[-2]                                                      | 
                                                                                                | 
                #pos in output tensor                                                           | 
                o = rows * out_strides[-2] + cols * out_strides[-1] + batch * out_strides[0]    | 
                #store                                                                          | 
                out[o] = acc                                                                    | 
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 1 parallel for-
loop(s) (originating from loops labelled: #5).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
 
---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
No allocation hoisting found
None





Task 3.4:

![3.4 Graph](threepointfourgraph.png)

/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 2 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/cudadrv/devicearray.py:888: NumbaPerformanceWarning: Host array used in CUDA kernel will incur copy overhead to/from device.
  warn(NumbaPerformanceWarning(msg))
Running size 64
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 8 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
{'fast': np.float64(0.007002830505371094), 'gpu': np.float64(0.010834217071533203)}
Running size 128
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 32 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
{'fast': np.float64(0.027009010314941406), 'gpu': np.float64(0.020455201466878254)}
Running size 256
{'fast': np.float64(0.164735476175944), 'gpu': np.float64(0.07354847590128581)}
Running size 512
{'fast': np.float64(1.3219787279764812), 'gpu': np.float64(0.20335078239440918)}
Running size 1024
{'fast': np.float64(8.680046240488688), 'gpu': np.float64(0.912109375)}

Timing summary
Size: 64
    fast: 0.00700
    gpu: 0.01083
Size: 128
    fast: 0.02701
    gpu: 0.02046
Size: 256
    fast: 0.16474
    gpu: 0.07355
Size: 512
    fast: 1.32198
    gpu: 0.20335
Size: 1024
    fast: 8.68005
    gpu: 0.91211


Task 3.5:
!cd $DIR; PYTHONPATH=/content/$DIR python3.12 project/run_fast_tensor.py --BACKEND gpu --HIDDEN 100 --DATASET split --RATE 0.05
Epoch  0  loss  8.103617195028985 correct 26
Epoch  10  loss  5.825488740964957 correct 42
Epoch  20  loss  4.188962668111611 correct 44
Epoch  30  loss  4.493786870889356 correct 46
Epoch  40  loss  4.348288489190148 correct 46
Epoch  50  loss  3.8993392712907724 correct 47
Epoch  60  loss  3.617435685106444 correct 49
Epoch  70  loss  2.179859853548842 correct 48
Epoch  80  loss  2.2084956516420586 correct 47
Epoch  90  loss  1.3085798324648525 correct 49
Epoch  100  loss  5.849945705500515 correct 43
Epoch  110  loss  2.942275399037138 correct 49
Epoch  120  loss  0.3749099148941203 correct 49
Epoch  130  loss  1.2067512213427938 correct 49
Epoch  140  loss  1.6143423535152561 correct 46
Epoch  150  loss  0.7105327166995582 correct 50
Epoch  160  loss  1.6758113819126197 correct 50
Epoch  170  loss  1.5734538990655198 correct 48
Epoch  180  loss  1.01398860173995 correct 50
Epoch  190  loss  2.525937550050106 correct 47
Epoch  200  loss  0.4186343602497854 correct 47
Epoch  210  loss  0.8403851157144074 correct 48
Epoch  220  loss  0.5911170553318343 correct 49
Epoch  230  loss  0.4235049837696682 correct 49
Epoch  240  loss  0.655561940298435 correct 49
Epoch  250  loss  0.916502881611854 correct 50
Epoch  260  loss  1.3640847145683244 correct 50
Epoch  270  loss  0.41679078140720477 correct 50
Epoch  280  loss  0.2804275942525735 correct 49
Epoch  290  loss  1.6434851028321993 correct 50
Epoch  300  loss  1.0725803516283732 correct 50
Epoch  310  loss  0.7118774266553561 correct 49
Epoch  320  loss  0.12816616181096624 correct 50
Epoch  330  loss  0.5659614828823417 correct 49
Epoch  340  loss  0.6554064732956527 correct 50
Epoch  350  loss  0.9117049969921138 correct 50
Epoch  360  loss  0.6666458346050442 correct 50
Epoch  370  loss  0.3757903231480096 correct 48
Epoch  380  loss  0.6207788465504561 correct 47
Epoch  390  loss  1.1534618724302284 correct 47
Epoch  400  loss  0.6183923731111534 correct 50
Epoch  410  loss  0.919749211386973 correct 50
Epoch  420  loss  1.1230646928237233 correct 47
Epoch  430  loss  1.1800670003811295 correct 49
Epoch  440  loss  0.3083388760863281 correct 50
Epoch  450  loss  0.46020219006305774 correct 50
Epoch  460  loss  0.6171931383621737 correct 50
Epoch  470  loss  0.3369694484094307 correct 49
Epoch  480  loss  1.0500714914082985 correct 50
Epoch  490  loss  2.0283058868132624 correct 44

