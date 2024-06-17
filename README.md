## Instructions 

   1. Write C++ file with function that has input/output types and calls cuda extension 
      
   2. Write a bridge pybind module to define a function in python that will invoke the above function 
      
   3. Build the above C++ file since C++ requires compiling and building before it can be called elsewhere. 
      1. We require `setup.py` in order to do this (Use template to write setup function) 
      2. Open terminal to current directory and write `pip install .` to build the C++ file 
      
   4. Create another `test.py` file to test code. Remember to import torch before cppcuda_tutorial
   
   5. Create the CUDA kernel `interpolation_kernel.cu` which has the actual C++ function that does the computation of the interpolation. 

   6. In the C++ bridge code (`interpolation.cpp`), you need to include a header file that declares all the cuda functions we write and exist so that these functions can be called by the bridge code file.  
   
      * A good practice to do this is creating a custom header file in `include` folder and writing all function definitions here (function definition has output type, function name and arguments)  

      * In bridge file, include the custom header file 
   
   7. Now, we edit the `setup.py` file since we are not building pure C++ code anymore. So instead of importing `CppExtension`, we import `CudaExtension` 

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

## The interpolation example  

The shapes of the inputs are: 

    feats: (N,8,F) 
        N = Number of cubes 
        8 = 8 vertices in cube 
        F = Number of features per vertex 
    
    point: (N,3) 
         N = Number of points 
        (Each corresponds to one cube in the feats tensor) 

        3 = 3D point coordinates 

We can parallelize this in two ways: 

1. **Parallelize `N`** : We can do this because interpolation of each point does not depend on another point 
   
2. **Parallelize `F`**: The interpolation of each feature doesn't depend on another feature, so we can parallelize that. 

