## Instructions 

   1. Write C++ file with function that has input/output types and calls cuda extension 
      
   2. Write a bridge pybind module to define a function in python that will invoke the above function 
      
   3. Build the above C++ file since C++ requires compiling and building before it can be called elsewhere. 
      1. We require `setup.py` in order to do this (Use template to write setup function) 
      2. Open terminal to current directory and write `pip install .` to build the C++ file 
      
   4. Create another `test.py` file to test code. Remember to import torch before cppcuda_tutorial 

## How CUDA Works 

   ! image ! 

   Host = CPU 
   Device = GPU 

   * When the host (or C++ program) calls a CUDA function, it instantiates a "kernel": one per function call. 

   * CUDA transfers data from cpu memory to gpu memory  

   * At the same time, a **grid** is generated, which is responsible for parallel computation. 
     * The grid contains of multiple **blocks** that are capable of parallel computation. The index/position of these blocks can be determined by the program. 

     * Each block has multiple **threads**, which is the smallest computation unit 

   * So a kernel initializes the above. Each thread can do its operation in parallel. 

   ### Example  

   For a vector addition problem where we want to add 2 matrices, we can make a kernel with as many threads as the number of elements in one of the vectors. Thus, we can do the computation of `a[i] + b[i]` by using the index position of the thread, since in matrix addition, we are adding elements corresponding to the same position in two vectors. Each thread performs one addition of two elements at its position.  

   ### FAQ

   1. Why can't we have many threads per grid? What's the use of blocks? 

      * Reason: **Hardware limitation** 
      * Max number of threads is limited to `1024` 
      * However, max number of blocks is: `(2^31 - 1) * 2^16`  
      * If we dont have blocks, we can only have a maximum of 1024 threads 


