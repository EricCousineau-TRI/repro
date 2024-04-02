import argparse
import os
import sys
import warnings

from jupyter import __file__ as _jupyter_file
from jupyter_core.command import main as _jupyter_main
from jupyterlab.labapp import main as _jupyterlab_main
from nbconvert.preprocessors import CellExecutionError, ExecutePreprocessor
# http://nbconvert.readthedocs.io/en/latest/execute_api.html
import nbformat

from runfiles import Rlocation


WORKSPACE_NAME = "jupyter_integration"

# TODO(eric.cousineau): Consider setting
# `MappingKernelManager.kernel_info_timeout` for `bazel test` running.


def jupyter_main(argv):
    """Replaces `sys.argv` with desired arguments, and runs Jupyter's main."""
    # TODO(eric.cousineau): Just use `execv`?
    if len(argv) > 0 and argv[0] == "lab":
        static_index = Rlocation(
            "pip_deps_jupyterlab/site-packages/jupyterlab/static/index.html"
        )
        app_dir = os.path.dirname(os.path.dirname(static_index))
        sys.argv = ["jupyter-lab"] + [f"--app-dir={app_dir}"] + argv[1:]
        print(sys.argv)
        return _jupyterlab_main()
    else:
        sys.argv = ["jupyter"] + argv
        return _jupyter_main()


def jupyter_is_interactive_run():
    """Returns true if being executed as an interactive notebook, false if as a
    test."""
    return os.environ.get("_JUPYTER_BAZEL_IS_TEST", "") != "1"


def _split_extra_args(argv):
    if "--" in argv:
        index = argv.index("--")
        extra = argv[index + 1:]
        argv = argv[:index]
        return argv, extra
    else:
        return argv, []


def _jupyter_bazel_notebook_main(notebook_path, argv):
    # This should *ONLY* be called by targets generated via `jupyter_py_*`
    # rules.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--notebook",
        action="store_true",
        help="Run using Jupyter Notebook",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run as a test (non-interactive)",
    )
    argv, jupyter_argv = _split_extra_args(argv)
    args = parser.parse_args(argv)

    if not args.test:
        print("Running notebook interactively")
        assert jupyter_is_interactive_run()
        if args.notebook:
            subcommand = "notebook"
        else:
            subcommand = "lab"
        # N.B. We use real path to appease the following
        # https://github.com/jupyter-server/jupyter_server/issues/711  # noqa
        notebook_path = os.path.realpath(notebook_path)
        exit(jupyter_main([subcommand] + jupyter_argv + [notebook_path]))
    else:
        print("Running notebook as a test (non-interactive)")
        os.environ["_JUPYTER_BAZEL_IS_TEST"] = "1"
        assert not jupyter_is_interactive_run()
        bazel_test_dir = os.environ.get("TEST_TMPDIR")
        if bazel_test_dir is not None:
            # Change IPython directory to use test directory.
            config_dir = os.path.join(bazel_test_dir, "jupyter")
            os.environ["IPYTHONDIR"] = config_dir
        # Escalate warnings for non-writable directories for settings
        # directories.
        warnings.filterwarnings(
            "error", message="IPython dir", category=UserWarning)
        # Execute using a preprocessor, rather than calling
        # `jupyter nbconvert`, as the latter writes an unused file to
        # `runfiles`.
        with open(notebook_path, encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=4)
        # Ensure that we use the notebook's directory, since that is used for
        # interactive sessions.
        notebook_dir = os.path.dirname(notebook_path)
        # TODO(eric.cousineau): See if there is a way to redirect this to
        # stdout, rather than having the notebook capture the output.
        ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
        try:
            ep.preprocess(nb, resources={'metadata': {'path': notebook_dir}})
        except CellExecutionError as e:
            if "\nSystemExit: 0\n" in e.traceback:
                print("Exited early")
            else:
                raise
        print("[ Done ]")
