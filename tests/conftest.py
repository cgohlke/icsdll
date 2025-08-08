# icsdll/tests/conftest.py

import os
import sys

if os.environ.get('VSCODE_CWD'):
    # work around pytest not using PYTHONPATH in VSCode
    sys.path.insert(
        0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    )


def pytest_report_header(config):
    try:
        from icsdll import __version__
        from numpy import __version__ as numpy

        return f'versions: icsdll-{__version__},  numpy-{numpy}'
    except Exception:
        pass


collect_ignore = ['_tmp', 'data']
