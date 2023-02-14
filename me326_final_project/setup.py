from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

d = generate_distutils_setup(
    packages=['me326_final_project'],
    scripts=['scripts'],
    package_dir={'': 'src'}
)

setup(**d)

