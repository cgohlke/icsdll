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
ONEAPI_ROOT = os.environ['ONEAPI_ROOT']

if (
    sys.platform != 'win32'
    or ('64 bit' not in sys.version)
    or sys.version_info[0] < 3
    or sys.version_info[1] < 8
):
    raise RuntimeError(PACKAGE + ' requires 64-bit Python>=3.8 for Windows')


def search(pattern, code, flags=0):
    # return first match for pattern in code
    match = re.search(pattern, code, flags)
    if match is None:
        raise ValueError(f'{pattern!r} not found')
    return match.groups()[0]


with open(PACKAGE + '/icsdll.py') as fh:
    code = fh.read()

version = search(r"__version__ = '(.*?)'", code)

description = search(r'"""(.*)\.(?:\r\n|\r|\n)', code)

readme = search(
    r'(?:\r\n|\r|\n){2}"""(.*)"""(?:\r\n|\r|\n){2}__version__',
    code,
    re.MULTILINE | re.DOTALL,
)
readme = '\n'.join(
    [description, '=' * len(description)] + readme.splitlines()[1:]
)


if 'sdist' in sys.argv:
    # update LICENSE and README files

    with open('README.rst', 'w') as fh:
        fh.write(readme)

    license = search(
        r'(# Copyright.*?(?:\r\n|\r|\n))(?:\r\n|\r|\n)+""',
        code,
        re.MULTILINE | re.DOTALL,
    )
    license = license.replace('# ', '').replace('#', '')

    with open('LICENSE', 'w') as fh:
        fh.write('BSD 3-Clause License\n\n')
        fh.write(license)

else:
    # build DLL

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
        extra_preargs=[
            '/EHsc',
            '/FD',
            '/DICS_EXPORTS',
            '/DICS_SIMFCS',
            '/openmp',
        ],
        include_dirs=['ics', ONEAPI_ROOT + '/mkl/latest/include'],
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
            ONEAPI_ROOT + '/mkl/latest/lib/intel64',
        ],
    )

setup(
    name=PACKAGE,
    version=version,
    description=description,
    long_description=readme,
    author='Christoph Gohlke',
    author_email='cgohlke@cgohlke.com',
    url='https://www.cgohlke.com',
    project_urls={
        'Bug Tracker': 'https://github.com/cgohlke/icsdll/issues',
        'Source Code': 'https://github.com/cgohlke/icsdll',
        # 'Documentation': 'https://',
    },
    python_requires='>=3.8',
    install_requires=['numpy'],
    setup_requires=['setuptools>=19.0'],
    packages=[PACKAGE],
    package_data={PACKAGE: [f'{DLLNAME}.dll', '*.dll']},
    libraries=[('', {'sources': []})],  # sets ispurelib = False
    license='BSD',
    zip_safe=False,
    platforms=['Windows'],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: BSD License',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'Operating System :: Microsoft :: Windows',
        'Programming Language :: C',
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
)
