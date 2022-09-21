import cupy as array_lib

import matplotlib.pyplot
from functools import wraps
import sys
__all__ = ["plt", "plot_wrapper"]


class plot_wrapper:
    def __init__(self, type_array_lib, to_numpy, str_array_lib=False):
        self.type_array_lib = type_array_lib
        self.to_numpy = to_numpy

        self.str_array_lib = str_array_lib
        self.is_lib_imported = str_array_lib if isinstance(str_array_lib, bool) else (str_array_lib in sys.modules.keys())

    def __getattr__(self, attr):
        func_to_be_wrapped = getattr(matplotlib.pyplot, attr)
        @wraps(func_to_be_wrapped)
        def _wrapping(*args, **kwargs):
            if self.is_lib_imported: #or try/exception #arr.get() instead of to_numpy(arr)
                args = [self.to_numpy(arr) if isinstance(arr, self.type_array_lib) else arr \
                        for arr in args]
                kwargs = {key:self.to_numpy(value) if isinstance(value, self.type_array_lib) else value \
                         for key, value in kwargs.items()}
            return func_to_be_wrapped(*args, **kwargs)
        return _wrapping


array_type = type(array_lib.empty(0))

def choose_array_to_numpy_call(library_name):
    return {
        "cupy": getattr(array_type, "get"),
        "torch": lambda x: getattr(array_type, "cpu") (getattr(array_type, "detach")(x)),
    }[library_name]

array_to_numpy_call = choose_array_to_numpy_call(array_lib.__name__)
plt = plot_wrapper(array_type, array_to_numpy_call, array_lib.__name__)