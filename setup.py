import glob
import os.path as osp
from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension


# ROOT_DIR = osp.dirname(osp.abspath(__file__))
# include_dirs = [osp.join(ROOT_DIR, "include")]

# sources = glob.glob('*.cpp')+glob.glob('*.cu')


setup(
    name='cppcuda_tutorial',
    version='1.0',
    author='Lakshya Gupta',
    author_email='lakshya.officialcanada@gmail.com',
    description='cppcuda_tutorial',
    long_description='cppcuda_tutorial',
    ext_modules=[
        CppExtension(
            name='cppcuda_tutorial',
            sources=['interpolation.cpp']
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
) 

# name: name of the package (example pytorch is called torch) 
# ext_module: The code you want to build (sources) 
# cmdclass tells we are BUILDING the code  
