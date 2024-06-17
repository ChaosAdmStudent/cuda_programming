## Instructions 

1. Write C++ file with function that has input/output types and calls cuda extension 
2. Write a bridge pybind module to define a function in python that will invoke the above function 
3. Build the above C++ file since C++ requires compiling and building before it can be called elsewhere. 
   1. We require `setup.py` in order to do this (Use template to write setup function) 
   2. Open terminal to current directory and write `pip install .` to build the C++ file 
4. Create another `test.py` file to test code. Remember to import torch before cppcuda_tutorial 