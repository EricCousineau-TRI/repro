import os
import subprocess
import sys
import warnings

from jupyter import __file__ as jupyter_file
from jupyter_core.command import main as _jupyter_main
# http://nbconvert.readthedocs.io/en/latest/execute_api.html
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor, CellExecutionError

WORKSPACE_NAME = "jupyter_integration"

# TODO(eric.cousineau): Consider setting
# `MappingKernelManager.kernel_info_timeout` for `bazel test` running.


def _prepend_path(key, p):
    # Prepends a path `p` into environment variable `key`.
    os.environ[key] = p + ":" + os.environ.get(key, '')


def jupyter_main(argv):
    """Replaces `sys.argv` with desired arguments, and runs Jupyter's main."""
    # TODO(eric.cousineau): Just use `execv`?
    # N.B. The *appropriate* `jupyter` binary must be first on ${PATH}.
    sys.argv = ["jupyter"] + argv
    return _jupyter_main()


def jupyter_is_interactive_run():
    """Returns true if being executed via `./bazel-bin` or `bazel run`.
    Otherwise, implies it is being run via `bazel test`."""
    return "_JUPYTER_BAZEL_INTERACTIVE_RUNFILES_DIR" in os.environ


def _resolve_runfiles_dir():
    # TODO(eric.cousineau): Ensure notebooks can run in any directory.
    manifest_file = os.environ.get("RUNFILES_MANIFEST_FILE")
    if manifest_file is None:
        # This happens when run via `bazel run`. Assume current directory.
        return os.getcwd()
    # Running via `./bazel-bin`.
    assert manifest_file.endswith("_manifest")
    result = manifest_file[:-len("_manifest")]
    output = os.path.join(result, WORKSPACE_NAME)
    assert os.path.isdir(output), output
    return output


def _jupyter_bazel_notebook_main(cur_dir, notebook_file, use_lab):
    # This should *ONLY* be called by targets generated via `jupyter_py_*`
    # rules.
    assert len(sys.argv) == 1, "Cannot specify arguments"

    # Assume that (hopefully) the notebook neighbors this generated file.
    # Failure mode: If user puts a slash in the `name` which does not match the
    # notebook's location. This should be infrequent.
    notebook_path = os.path.join(cur_dir, notebook_file)
    if not os.path.isfile(notebook_path):
        # `name` may contain a subdirectory. Just use basename of file.
        notebook_path = os.path.join(cur_dir, os.path.basename(notebook_file))

    # If we can write to the notebook (e.g. `./bazel-bin` or `bazel run`), run
    # it interactively.
    is_interactive = os.access(notebook_path, os.W_OK)

    if is_interactive:
        print("Running notebook interactively (writeable)")
        # Ensure that we propagate the fact that this is interactive.
        runfiles_dir = _resolve_runfiles_dir()
        os.environ["_JUPYTER_BAZEL_INTERACTIVE_RUNFILES_DIR"] = runfiles_dir
        os.chdir(runfiles_dir)
        # Double-check public API.
        assert jupyter_is_interactive_run()
        if use_lab:
            subcommand = "lab"
        else:
            subcommand = "notebook"
        exit(jupyter_main([subcommand, notebook_path]))
    else:
        # WARNING: Fragile if used in composition (if one notebook spawns
        # another, for whatever reason). For now, do not worry about it.
        assert not jupyter_is_interactive_run()
        print("Running notebook non-interactively (read-only):")
        print("  {}".format(notebook_file))
        # Escalate warnings for non-writable directories for settings
        # directories.
        warnings.filterwarnings(
            "error", message="IPython dir", category=UserWarning)
        tmp_dir = os.environ.get("TEST_TMPDIR")
        if tmp_dir:
            # Change IPython directory to use test directory.
            config_dir = os.path.join(tmp_dir, "jupyter")
            os.environ["IPYTHONDIR"] = config_dir
        # Execute using a preprocessor, rather than calling
        # `jupyter nbconvert`, as the latter writes an unused file to
        # `runfiles`.
        with open(notebook_path) as f:
            nb = nbformat.read(f, as_version=4)
        # TODO(eric.cousineau): See if there is a way to redirect this to
        # stdout, rather than having the notebook capture the output.
        ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
        try:
            ep.preprocess(nb, {'metadata': {'path': os.getcwd()}})
        except CellExecutionError as e:
            if "\nSystemExit: 0\n" in e.traceback:
                print("Exited early")
            else:
                raise


def jupyter_bazel_notebook_init():
    """Call this at the very beginning of each notebook that you intend to use
    via Bazel.

    Jupyter by default sets the PWD of a new kernel to the directory of the
    current notebook, so this must be called at the start of each kernel
    session, hence it must be placed in a notebook's first few cells.
    This permits the notebook to run under interactive sessions and
    `bazel run` / `bazel test`.

    @note `--notebook-dir` / `NotebookApp.notebook_dir` only sets the initial
    tree's notebook directory, but not the kernel's PWD.
    """
    # TODO(eric.cousineau): Figure out how to make a notebook's kernel start at
    # a desired directory without this hackery.
    if jupyter_is_interactive_run():
        runfiles_dir = os.environ["_JUPYTER_BAZEL_INTERACTIVE_RUNFILES_DIR"]
        print("Setting PWD to runfiles: {}".format(runfiles_dir))
        os.chdir(runfiles_dir)
    else:
        print("Keeping Jupyter PWD: {}".format(os.getcwd()))
