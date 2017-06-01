from __future__ import print_function

# Unfortunately, numpy should be installed before
# Also, installation requires fortran + openmp.
from numpy.distutils.core import Extension, setup

setup(
    name="infiniteboost",
    version='0.1',
    description="InfiniteBoost ",
    long_description=""" See more details about algorithm in the paper about InfiniteBoost """,
    url='https://arogozhnikov.github.io',

    # Author details
    author='Alex Rogozhnikov, Likhomanenko Tatiana',

    # Choose your license
    license='Apache 2.0',
    packages=['infiniteboost'],

    classifiers=[
        # Indicate who your project is intended for
        'Intended Audience :: Science/Research',

        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: Apache Software License',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 2.7 ',
        'Programming Language :: Python :: 3.4 ',
    ],

    keywords='Machine Learning',

    ext_modules=[Extension(name='infiniteboost.fortranfunctions',
                           sources=['infiniteboost/fortranfunctions.f90'],
                           extra_link_args=["-lgomp"],
                           extra_f90_compile_args=["-fopenmp -O3"])],

    # List run-time dependencies here. These will be installed by pip when your project is installed.
    install_requires=[
        'numpy >= 1.12',
        'scipy >= 0.19',
        'pandas >= 0.19.2',
        'scikit-learn >= 0.18',
        'six',
        'hep_ml == 0.4',
        'nose'
    ],
)
