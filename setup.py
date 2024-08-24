"""
python -c "print()"

refs:
    - setup tools: https://setuptools.pypa.io/en/latest/userguide/package_discovery.html#using-find-or-find-packages
    - https://stackoverflow.com/questions/70295885/how-does-one-install-pytorch-and-related-tools-from-within-the-setup-py-install
"""
from setuptools import setup
from setuptools import find_packages
import os

here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='dataset-creation',  # project name
    version='0.0.1',
    description="Creating Data for the Alignment Project",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/iddahmlauzi/dataset-creation",
    author='Iddah Mlauzi',
    author_email='iddah@gmail.com',
    python_requires='>=3.11',
    license='Apache 2.0',

    # ref: https://chat.openai.com/c/d0edae00-0eb2-4837-b492-df1d595b6cab
    # The `package_dir` parameter is a dictionary that maps package names to directories.
    # A key of an empty string represents the root package, and its corresponding value
    # is the directory containing the root package. Here, the root package is set to the
    # 'src' directory.
    #
    # The use of an empty string `''` as a key is significant. In the context of setuptools,
    # an empty string `''` denotes the root package of the project. It means that the
    # packages and modules located in the specified directory ('src' in this case) are
    # considered to be in the root of the package hierarchy. This is crucial for correctly
    # resolving package and module imports when the project is installed.
    #
    # By specifying `{'': 'src'}`, we are informing setuptools that the 'src' directory is
    # the location of the root package, and it should look in this directory to find the
    # Python packages and modules to be included in the distribution.
    package_dir={'': 'src'},

    # The `packages` parameter lists all Python packages that should be included in the
    # distribution. A Python package is a way of organizing related Python modules into a
    # directory hierarchy. Any directory containing an __init__.py file is considered a
    # Python package.
    #
    # `find_packages('src')` is a convenience function provided by setuptools, which
    # automatically discovers and lists all packages in the specified 'src' directory.
    # This means it will include all directories in 'src' that contain an __init__.py file,
    # treating them as Python packages to be included in the distribution.
    #
    # By using `find_packages('src')`, we ensure that all valid Python packages inside the
    # 'src' directory, regardless of their depth in the directory hierarchy, are included
    # in the distribution, eliminating the need to manually list them. This is particularly
    # useful for projects with a large number of packages and subpackages, as it reduces
    # the risk of omitting packages from the distribution.
    packages=find_packages('py_src'),
    # When using `pip install -e .`, the package is installed in 'editable' or 'develop' mode.
    # This means that changes to the source files immediately affect the installed package
    # without requiring a reinstall. This is extremely useful during development as it allows
    # for testing and iteration without the constant need for reinstallation.
    #
    # In 'editable' mode, the correct resolution of package and module locations is crucial.
    # The `package_dir` and `packages` configurations play a vital role in this. If the
    # `package_dir` is incorrectly set, or if a package is omitted from the `packages` list,
    # it can lead to ImportError due to Python not being able to locate the packages and
    # modules correctly.
    #
    # Therefore, when using `pip install -e .`, it is essential to ensure that `package_dir`
    # correctly maps to the root of the package hierarchy and that `packages` includes all
    # the necessary packages by using `find_packages`, especially when the project has a
    # complex structure with nested packages. This ensures that the Python interpreter can
    # correctly resolve imports and locate the source files, allowing for a smooth and
    # efficient development workflow.


# for pytorch see doc string at the top of file
    install_requires=[
        'fire',
        'dill',
        'networkx>=2.5',
        'scipy',
        'scikit-learn',
        'lark-parser',
        'tensorboard',
        'pandas',
        'progressbar2',
        'requests',
        'aiohttp',
        'numpy<2',
        'plotly',
        'wandb',
        'matplotlib',
        'nvidia-htop',
        'openai',
        'anthropic',
        'jsonlines',
   
        'twine',
       
        # ampere
        'dspy-ai',
        'torch==2.2.1',  # 2.2 net supported due to vllm see: https://github.com/vllm-project/vllm/issues/2747
        'transformers>=4.40',
        'accelerate==0.29.2',
        'datasets==2.14.7',  
        'evaluate==0.4.1', 
        'bitsandbytes== 0.43.0',
        'einops',
        'flash-attn>=2.5.8',
        'vllm==0.4.1', # my gold-ai-olympiad project uses 0.4.0.post1 ref: https://github.com/vllm-project/vllm/issues/2747
        
        'zstandard',

    ]
)
