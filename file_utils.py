import sys
import os
from pathlib import Path

def get_bundle_filepath(filepath: str) -> str:
    if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
        bundle_dir = str(sys._MEIPASS)
        return os.path.join(Path.cwd(), bundle_dir, filepath)
    else:
        return os.path.join(Path.cwd(), filepath)