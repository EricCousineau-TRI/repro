"""
The purpose of these macros are to:

* Minimize the chance of accidental One-Definition-Rule
(https://en.wikipedia.org/wiki/One_Definition_Rule) violations.
    * Example: Mixing static libraries like `@drake//multibody/plant` with
    `@drake//:drake_shared_library` in a target's `deps`.
* Provide linting rules to discourage the use of `pybind_py_library`
(where these rules are not enforced).

The two primary macros of interest are `anzu_cc_shared_library`, which guides
the creation of shared libraries to re-incorporate Anzu C++ code in a form that
Python can consume, and `anzu_pybind_py_library`, where the bindings are
written.

The shared library should go under the given package with the name
`:shared_library`, and the bindings should be written as the target `cc_py`
(source file: `cc_py.cc`), and should be imported as `anzu.{package}.cc`. This
is done to make the intent clear of the module (they are C++ bindings) and
make nested-package bindings unambiguous.

As an example, see `//common:cc_py`.

**TODO(eric)**: Show nuances of including both C++ and Python deps for packages
downstream of `//common`.

**TODO(eric)**: Consider place Python bindings in separate package if we have a
good mechanism for it.
"""

load(
    "//tools/skylark:anzu_cc.bzl",
    "anzu_cc_binary",
    "anzu_cc_library",
)
load(
    "@drake//tools/skylark:pybind.bzl",
    "pybind_py_library",
)
load("//tools/skylark:anzu_py.bzl", "anzu_py_test")

# These should be libraries that are purely for C++ (do not need Python).
# Do not edit this without <insert people>'s approval.
_CC_DEP_ALLOWLIST = [
    ":example_odr_robust_library",
    "//lcmtypes:lcmtypes_anzu_cc",
    "@bazel_tools//tools/cpp/runfiles",
    "@boost//:boost_headers",
    "@common_robotics_utilities//:common_robotics_utilities_headers_only",
    "@drake//:drake_shared_library",
    "@eigen",
    "@fmt",
    "@opencv",
    # N.B. It should be OK to statically link this multiple times.
    "@tinyobjloader",
]

# These should be libraries that are only intended to be used with Python.
# Do not edit this without <insert people>'s approval.
_PYBIND_CC_DEP_ALLOWLIST = [
    "@pybind11",
    "@python",
    "@numpy",
]

def anzu_pybind_bazel_lint(
        name = "anzu_pybind_bazel_lint",
        exclude = [],
        extra_srcs = []):
    """Adds lint rules that scans Skylark code and fails if there are any
    direct usages of `pybind_py_library` or `drake_pybind_py_library`."""
    files = extra_srcs + native.glob([
        "*.bzl",
        "*.BUILD.bazel",
        "BUILD.bazel",
    ], exclude = exclude)
    anzu_py_test(
        name = name,
        data = files,
        args = ["$(location {})".format(x) for x in files],
        srcs = ["//tools/lint:anzu_pybind_bazel_check.py"],
        deps = ["//tools/lint:anzu_pybind_bazel_check"],
        tags = ["lint"],
    )

def _valid_cc_dep(cc_dep):
    if cc_dep in _CC_DEP_ALLOWLIST:
        return True
    if cc_dep.endswith(":shared_library"):
        return True
    return False

def _check_cc_deps(cc_deps):
    # Reject any C++ dependencies that may violate ODR.
    bad_cc_deps = []
    for cc_dep in cc_deps:
        if not _valid_cc_dep(cc_dep):
            bad_cc_deps.append(cc_dep)
    if bad_cc_deps:
        fail("The following C++ dependencies are invalid due to potential " +
             "ODR violations: " + str(bad_cc_deps))

def _check_pybind_cc_deps(cc_deps):
    # Reject any pyinbd11 dependencies that may violate ODR.
    bad_cc_deps = []
    for cc_dep in cc_deps:
        if _valid_cc_dep(cc_dep):
            continue
        if cc_dep in _PYBIND_CC_DEP_ALLOWLIST:
            continue
        if cc_dep.endswith("_pybind"):
            continue
        bad_cc_deps.append(cc_dep)
    if bad_cc_deps:
        fail("The following pybind11 C++ dependencies are invalid due to " +
             "potential ODR violations: " + str(bad_cc_deps))

def anzu_pybind_py_library(
        name,
        cc_deps = [],
        **kwargs):
    """Macro that wraps `pybind_py_library` to compile pybind11 C++ code to
    expose bindings to Python.

    This macro will throw an error if `cc_deps` contain anything that may
    violate ODR.
    """
    _check_pybind_cc_deps(cc_deps)
    pybind_py_library(
        name = name,
        cc_deps = cc_deps,
        **kwargs
    )

def anzu_cc_shared_library(
        name,
        copts = [],
        hdrs = [],
        srcs = [],
        deps = [],
        **kwargs):
    """Creates a shared library, primarily for usage with Python bindings.
    `name` must be `shared_library`.

    For simplicity, this should generally not contain any Python-interpreter
    dependent libraries; this should only deal with re-wrapping existing C++
    code into a library, while keeping dependencies at a bare minimum.

    @note This macro is constructed such that `deps` should be
    ODR-violation-robust, e.g. all are shared library dependencies,
    header-only, or have no global state and will not encounter RTTI visibility
    issues. This will throw an error if any suspiciuos dependencies are added.
    """
    if name != "shared_library":
        fail("This library *must* be named `shared_library`. " +
             "Please review `python-conventions.md`.")
    _check_cc_deps(deps)

    friendly_package_name = native.package_name().replace("/", ".")
    so_name = "anzu_{}_shared_library".format(friendly_package_name)
    solib = "lib{}.so.1".format(so_name)

    # Create main shared library.
    anzu_cc_binary(
        name = solib,
        srcs = srcs + hdrs,
        linkshared = 1,
        linkstatic = 1,
        deps = deps,
        **kwargs
    )

    # Expose shared library and headers for transitive dependencies.
    anzu_cc_library(
        name = name,
        hdrs = hdrs,
        srcs = [solib],
        deps = deps,
        _skylark_internal_nested_call = True,
        **kwargs
    )

def anzu_pybind_cc_library(
        name,
        hdrs = [],
        srcs = [],
        deps = [],
        **kwargs):
    """Defines a anzu_cc_library that is fit for usage in Python C++ code used
    in anzu_pybind_py_library."""
    _check_pybind_cc_deps(deps)
    for hdr in hdrs:
        if not hdr.endswith("_pybind.h"):
            fail("Invalid hdr: " + hdr)
    for src in srcs:
        if not src.endswith("_pybind.cc"):
            fail("Invalid src: " + src)
    anzu_cc_library(
        name = name,
        hdrs = hdrs,
        srcs = srcs,
        deps = deps,
        _skylark_internal_nested_call = True,
        **kwargs
    )
