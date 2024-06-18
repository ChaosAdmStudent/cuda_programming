## Instructions + Explanation

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

   8. In the CUDA file in the function implementation, make sure there are 3 or less parallelizable dimensions. Now, create a `dim3` variable and set number of threads per dimension. (`const dim3 threads(16,16`) [ Since we want 256 total threads evenly distributed in 2 dimensions. sqrt(256) = 16]. Experiment with this according to balance of parallelizable dimensions (N and F here) and the computation complexity on different dimensions 

   9. * The output is created inside the cuda function as empty tensor which will be filled within the kernel function later.  

      * Before launching kernel, we specify the number of threads and blocks 
   
   

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
   
   2. What is `AT_DISPATCH_FLOATING_TYPES`? 

      * Even though Pytorch abstracts away the data type for us during code writing, behind the scenes it still has a concrete data type for the function output and it needs to call the function with the correct data type signature. This is what this function does. It takes in the type that it is analyzing as the first argument. After that, it's conceptually working like a switch-case statement. It instantiates the kernel with the correct type signature.

      ```
      switch (tensor.type().scalarType()) 
      {
         case torch::ScalarType::Double:
            return function<double>(tensor.data<double>());

         case torch::ScalarType::Float:
            return function<float>(tensor.data<float>());

         ...
      }
      ``` 

      * This function takes 3 inputs: a type, a name (for error msgs) and a lambda function. There is a `scalar_t` available inside the lamdba function which is defined as the type that the tensor actually is at runtime in that context. An easy way to call the right CUDA kernel is to make the CUDA kernel a template function with the `scalar_t` alias. 
   
   3. Explain the function content of `AT_DISPATCH_FLOATING_TYPES`? 

      * `trilinear_fw_kernel` is the kernel we want to call. 

      * `scalar_t` is the template for the data type. It allows kernel to do computation for different data types. 

      * The kernel call is in the format `kernel<<<blocks,threads>>>(input1, input2, input3, ...)` . In this case, we give 3 inputs to the kernel: feats, points and the empty output tensor. 

      * The kernel NEVER returns anything. All operations are done in place. That's why we always provide an output tensor that is filled by the kernel computation. 

      * The `.packed_accessor` helps convert the "tensor" type to a type that CUDA recognizes. Without it, it wouldn't know what `torch::Tensor feats` is.  

      * The arguments of `packed_accessor` are: 
        * The first one sets the data type of the corresponding tensor. Because we used AT_DISPATCH_FLOATING_TYPES, we can just set this equal to `scalar_t` so that it automatically sets data type of the tensor to the corresponding detected floating data type (float32 or float64)  

        * The 2nd argument means the number of dimensions of the tensor. Example: feats: (N,8,F). Hence feats has 3 dimensions. Similarly, points: (N,3), hence it has 2 dimensions etc. 

        * The 3rd argument `RestrictPtrTraits` ensures that "feats" (or any other tensor) doesn't overlay with any other tensors.

        * The 4th argument is the type used for indexing and offset computations. Here, we use `size_t` which is an unsigned integer type. It's used here because tensor indices and offsets can't be negative, and size_t can hold very large values  

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

## How to decide how many threads to create for a kernel? 

   For this, inspect algorithm and see how many dimensions are there with parallel computation. In the interpolation example above, we have 2 dimensions (N and F).  

   * Threads can have atmost 3 dimensions (hardware limitation). If your algo has more than 3 dimensions of parallelization, redesign the algorithm. 

## How to compute number of blocks? 

   < insert image here from tab > 

   The number of blocks for each dimension comes from this formula: 

   `const dim3 blocks((N+threads.x-1)/threads.x, (F+threads.y-1)/threads.y);` 