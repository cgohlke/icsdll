# icsdll/setup.py

"""Icsdll package setuptools script."""

import os
import sys
import re
import glob

from setuptools import setup

from distutils import ccompiler

PACKAGE = 'icsdll'
DLLNAME = 'ICSx64'
INTELDIR = (
    'C:/Program Files (x86)/IntelSWTools/compilers_and_libraries/windows'
)

if (
    sys.platform != 'win32'
    or ('64 bit' not in sys.version)
    or sys.version_info[0] < 3
    or sys.version_info[1] < 7
):
    raise RuntimeError(PACKAGE + ' requires 64-bit Python>=3.7 for Windows')

with open(PACKAGE + '/icsdll.py') as fh:
    code = fh.read()

version = re.search(r"__version__ = '(.*?)'", code).groups()[0]

description = re.search(r'"""(.*)\.(?:\r\n|\r|\n)', code).groups()[0]

readme = re.search(
    r'(?:\r\n|\r|\n){2}"""(.*)"""(?:\r\n|\r|\n){2}__version__',
    code,
    re.MULTILINE | re.DOTALL,
).groups()[0]

readme = '\n'.join(
    [description, '=' * len(description)] + readme.splitlines()[1:]
)

license = re.search(
    r'(# Copyright.*?(?:\r\n|\r|\n))(?:\r\n|\r|\n)+""',
    code,
    re.MULTILINE | re.DOTALL,
).groups()[0]

license = license.replace('# ', '').replace('#', '')

if 'sdist' in sys.argv:
    with open('LICENSE', 'w') as fh:
        fh.write(license)
    with open('README.rst', 'w') as fh:
        fh.write(readme)

for f in glob.glob(os.path.join(PACKAGE, DLLNAME + '.*')):
    try:
        os.remove(f)
    except Exception:
        pass

sources = [
    os.path.join('ics', c)
    for c in ('dllmain.cpp', 'ics.cpp', 'rfft.cpp', 'yxt.cpp', 'nlsp.cpp')
]

compiler = ccompiler.new_compiler()
objects = compiler.compile(
    sources,
    'build',
    extra_preargs=['/EHsc', '/FD', '/DICS_EXPORTS', '/openmp', '/DICS_SIMFCS'],
    include_dirs=['ics', INTELDIR + '/mkl/include'],
)

compiler.link_shared_lib(
    objects,
    os.path.join(PACKAGE, DLLNAME),
    extra_preargs=['/DLL'],
    libraries=[
        'mkl_core',
        'mkl_intel_lp64',
        'mkl_sequential',
        # 'mkl_intel_thread',
        # 'libiomp5md',
    ],
    library_dirs=[
        INTELDIR + '/mkl/lib/intel64_win',
        INTELDIR + '/compiler/lib/intel64_win',
    ],
)

setup(
    name=PACKAGE,
    version=version,
    description=description,
    long_description=readme,
    author='Christoph Gohlke',
    author_email='cgohlke@uci.edu',
    url='https://www.lfd.uci.edu/~gohlke/',
    project_urls={
        'Bug Tracker': 'https://github.com/cgohlke/icsdll/issues',
        'Source Code': 'https://github.com/cgohlke/icsdll',
        # 'Documentation': 'https://',
    },
    python_requires='>=3.7',
    install_requires=['numpy>=1.19.1'],
    setup_requires=['setuptools>=18.0'],
    packages=[PACKAGE],
    package_data={PACKAGE: ['*.dll']},
    libraries=[('', {'sources': []})],  # sets ispurelib = False
    license='BSD',
    zip_safe=False,
    platforms=['Windows'],
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'License :: OSI Approved :: BSD License',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'Operating System :: Microsoft :: Windows',
        'Programming Language :: C',
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
)
