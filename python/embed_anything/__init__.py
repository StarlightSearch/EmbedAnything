import os
os.add_dll_directory(os.path.dirname(__file__)+os.path.sep + "lib")
from .embed_anything import *

__doc__ = embed_anything.__doc__
if hasattr(embed_anything, "__all__"):
    __all__ = embed_anything.__all__