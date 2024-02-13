import sys
from cx_Freeze import setup, Executable

base = 'Win32GUI' if sys.platform == 'win32' else None

includefiles = ['ffmpeg/', 'img/', 'bdetectionmodel_05_01_23.onnx']
includes = []
excludes = ['Tkinter']
packages = ['moviepy']

setup(
    name='AutoComper',
    version='1.0.2',
    description='Automatic Comp Creation Tool',
    author='wz-bff',
    options={'build_exe': {'includes': includes, 'excludes': excludes,
                           'packages': packages, 'include_files': includefiles}},
    executables=[Executable('autocomper.py',
                            base=base,
                            icon="app.ico")]
)
