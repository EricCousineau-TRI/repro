"""
Import this first, before *anything* else, if you are using `multiprocessing`
directly or indirectly (e.g. `parallel_work`) and you encounter "freezing".

This should *only* be used by the *main* module.

For more information, see:
https://stackoverflow.com/questions/17053671/python-how-do-you-stop-numpy-from-multithreading  # noqa
"""
import importlib
import inspect
import os
import sys

_IGNORE_CALLING_FILENAMES = {
    # Skip importlib modules, as they're used to parse and execute the code.
    '<frozen importlib._bootstrap_external>',
    '<frozen importlib._bootstrap>',
    importlib.__file__,
}


def _get_calling_module_name():
    calling_stack = inspect.stack()[3:]
    for frame_info in calling_stack:
        if frame_info.filename not in _IGNORE_CALLING_FILENAMES:
            break
    else:
        assert False, "Stack too shallow?"
    m = inspect.getmodule(frame_info.frame)
    return m.__name__


def _set_env():
    calling_name = _get_calling_module_name()
    assert calling_name == "__main__", (
        f"This should only be imported when the calling module is '__main__', "
        f"but it is '{calling_name}'"
    )

    this_should_be_imported_before = {"cv2", "numpy"}

    toplevel_modules = set([name for name in sys.modules if "." not in name])
    already_imported = this_should_be_imported_before & toplevel_modules
    assert len(already_imported) == 0, (
        f"Modules should be imported *after* '{__name__}': {already_imported}"
    )

    os.environ.update(
        OMP_NUM_THREADS='1',
        OPENBLAS_NUM_THREADS='1',
        NUMEXPR_NUM_THREADS='1',
        MKL_NUM_THREADS='1',
    )

    import numpy as np  # noqa
    import cv2  # noqa
    cv2.setNumThreads(0)


_set_env()
