"""
The purpose of these macros are to:

* Minimize the chance of accidental One-Definition-Rule
(https://en.wikipedia.org/wiki/One_Definition_Rule) violations.
    * Example: Mixing static libraries like `@drake//multibody/plant` with
    `@drake//:drake_shared_library` in a target's `deps`.
* Provide linting rules to discourage the use of `pybind_py_library`
(where these rules are not enforced).

For an example of what ODR violations like in "minimal" C++ code:
https://github.com/EricCousineau-TRI/repro/tree/cbac41b9e2/bazel/bazel_shared_lib_odr_instrumentation

Search "ODR" in Drake to find out more there.

The two primary macros of interest are `anzu_cc_shared_library`, which guides
the creation of shared libraries to re-incorporate Anzu C++ code in a form that
Python can consume, and `anzu_pybind_py_library`, where the bindings are
written.

The shared library should go under the given package with the name
`:shared_library`, and the bindings should be written as the target `cc_py`
(source file: `cc_py.cc`), and should be imported as `anzu.{package}.cc`. This
is done to make the intent clear of the module (they are C++ bindings) and
make nested-package bindings unambiguous.

Third party libraries that need to be re-linked should use
`anzu_third_party_cc_shared_library`.

For an in-the-wild example, see `//common:shared_library` and
`//common:cc_py`.

For an-place, slightly non-trivial example:

    load(
        "//tools:anzu_pybind.bzl",
        "anzu_cc_shared_library",
        "anzu_pybind_py_library",
    )

    anzu_cc_shared_library(
        name = "shared_library",
        package_deps = [
            ":my_system",
            ":my_geometry_thing",
        ],
        deps = [
            "//common:shared_library",
            "@drake//:drake_shared_library",
            "@tinyobjloader",
        ],
    )

    anzu_pybind_py_library(
        name = "cc_py",
        cc_deps = [
            ":shared_library",
            "@drake//bindings/pydrake/common:cpp_param_pybind",
            "@drake//bindings/pydrake/common:cpp_template_pybind",
            "@drake//bindings/pydrake/common:default_scalars_pybind",
            "@drake//bindings/pydrake/common:value_pybind",
        ],
        cc_so_name = "cc",
        cc_srcs = ["cc_py.cc"],
        py_deps = [
            ":module_py",
            "//common:cc_py",
            "@drake//bindings/pydrake",
        ],
    )
"""  # noqa

load(
    "@drake//tools/skylark:pybind.bzl",
    "pybind_py_library",
)
load(
    ":extract_cc.bzl",
    "extract_cc_object_files_ctx",
    "extract_package_cc_hdrs_srcs_data",
)
load(
    ":extract_deps.bzl",
    "ExtractDepsInfo",
    "aggregate_extract_deps",
    "expand_label_deps",
    "extract_deps_aspect",
    "remap_label_for_anzu",
)
load(
    ":labels.bzl",
    labels_mod = "labels",
)
load(
    ":pprint_ish.bzl",
    "pformat_ish",
    "pprint_ish",
)
load(
    ":containers.bzl",
    "combine_and_sort",
    "indent",
    "set_diff",
    "sort_friendly",
    "to_str_list",
    "uniq",
)

# HACK
anzu_cc_binary = native.cc_binary
anzu_cc_library = native.cc_library
anzu_py_library = native.py_library
anzu_py_test = native.py_test

# [ Manifest and Convention Information ]

# These are libraries for use with pure C++ shared libraries (not consuming
# Python).
# These should be libraries that are not only safe to use for C++ shared
# libraries, but also must be forwarded as the dependencies of said shared
# library.
# Please note that this should also permit transitive deps, e.g. we don't have
# to explicitly list out all of Drake's dependencies since we list it here.

_CC_DEP_ALLOWLIST = [
    Label(x)
    for x in [
        # *shared_library handled by  _valid_cc_dep().
        "@boost//:boost_headers",
        "@drake//:drake_shared_library",
    ]
]

# Please be very conservative about this list.
_CC_DEP_ALLOW_LIST_REPOSITORY = [
    "@ros2",
]

def _valid_cc_dep(dep_label):
    if dep_label.name.startswith("shared_library"):
        return True
    elif dep_label.name.endswith("_ros_msgs_cc"):
        return True
    elif _in_repositories(dep_label, _CC_DEP_ALLOW_LIST_REPOSITORY):
        # All of these tend to be shared libs.
        return True
    elif dep_label in _CC_DEP_ALLOWLIST:
        return True
    return False

def _valid_shared_library_name(name):
    return name.startswith("shared_library")

def _assert_valid_shared_library_name(name):
    if not _valid_shared_library_name(name):
        fail("This library *must* start with `shared_library`. " +
             "Please review `anzu_pybind.bzl`.")

# These are workspaces that use `anzu_*_cc_shared_library`. Be sure to
# update `_remap_to_shared_deps_from_single_label` to acommodate any remapping.
_WORKSPACES_WITH_ANZU_SHARED_LIBRARY = [
    "",
]

def _remap_to_shared_deps_from_single_label(
        ctx,
        dep_label,
        cc_dep_allowlist_expanded,
        static_ignorable):
    remapped = []
    maybe_invalid = []
    if static_ignorable and _ignorable_static_cc_dep(dep_label):
        pass
    elif _valid_cc_dep(dep_label):
        # Fast lookup.
        remapped = [dep_label]
    elif labels_mod.in_current_package(dep_label, ctx):
        # We're going to relink / recompile.
        pass
    elif labels_mod.in_root_workspace(dep_label):
        # Another package; assume shared library + Python.
        qualified_package = labels_mod.qualify_package(dep_label)
        remapped = [Label(qualified_package + ":shared_library")]
    elif labels_mod.in_workspace(dep_label, "drake"):
        # Drake generally means shared library.
        remapped = [Label("@drake//:drake_shared_library")]
    elif dep_label in cc_dep_allowlist_expanded:
        # TODO(eric.cousineau): Do we need to defer transitive check towards
        # the end?
        remapped = [dep_label]
    elif _in_our_workspaces(dep_label):
        fail("This should have a remap rule: " + pformat_ish(dep_label))
    else:
        # Unknown mapping; either invalid or needs to be added in manifeset.
        maybe_invalid = [dep_label]
    return remapped, maybe_invalid

# These should be libraries that are only intended to be used with Python.
# Do not edit this without Eric, Jeremy, or Sam's approval.
_PYBIND_CC_DEP_ALLOWLIST = [
    Label(x)
    for x in [
        "@pybind11",
        "@python",
        "@numpy",
    ]
]

def _valid_pybind_cc_dep(dep_label):
    if _valid_cc_dep(dep_label):
        return True
    elif dep_label in _PYBIND_CC_DEP_ALLOWLIST:
        return True
    elif dep_label.name.endswith("_pybind"):
        return True
    return False

# The below "transitive ignorable" can be ignored only iff they come up in
# analysis through anayis of dependencies for
# `anzu_*_cc_shared_library(package_deps)` (static deps).
# At present, they should trigger an error if added in
# `anzu_*_cc_shared_library(deps)` (shared deps).
# TODO(eric.cousineau): This feels messy...

_CC_DEP_STATIC_IGNORABLE = [
    Label(x)
    for x in [
        "@bazel_tools//tools/cpp/runfiles",
        "@ccd",
        "@common_robotics_utilities//:common_robotics_utilities_headers_only",
        "@csdp",
        "@gfortran//:runtime",
        "@ghc_filesystem",
        "@glib",
        "@gz_math_internal//:gz_math",
        "@gz_utils_internal//:gz_utils",
        "@libblas",
        "@liblapack",
        "@msgpack",
        "@osqp",
        "@petsc",
        "@qdldl",
        "@scs//:scsdir",
        "@sdformat_internal//:sdformat",
        "@snopt//:fortran_objects",
        "@snopt//:snopt_cwrap",
        "@stduuid",
        "@suitesparse//:amd",
        "@usockets",
        "@uwebsockets",
        "@yaml_cpp_internal//:yaml_cpp",
    ]
]

_CC_DEP_STATIC_IGNORABLE_REPOSITORY = [
    "@abseil_cpp_internal",
    "@conex",
    "@fcl",
    "@nlopt_internal",
]

def _ignorable_static_cc_dep(dep_label):
    if dep_label in _CC_DEP_STATIC_IGNORABLE:
        return True
    if _in_repositories(dep_label, _CC_DEP_STATIC_IGNORABLE_REPOSITORY):
        return True
    return False

# [ Nominal starlark here ]

def _in_repositories(label, repositories):
    for repository in repositories:
        workspace = labels_mod.strip_prefix("@", repository)
        if labels_mod.in_workspace(label, workspace):
            return True
    return False

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

def _assert_valid_pybind_cc_deps(cc_deps):
    # Reject any pyinbd11 dependencies that may violate ODR.
    bad_cc_deps = []
    for cc_dep in cc_deps:
        cc_dep_label = labels_mod.to_label(cc_dep)
        if not _valid_pybind_cc_dep(cc_dep_label):
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
    _assert_valid_pybind_cc_deps(cc_deps)
    pybind_py_library(
        name = name,
        cc_deps = cc_deps,
        **kwargs
    )

def _assert_valid_package_deps(package_deps):
    for dep in package_deps:
        dep_label = labels_mod.to_label(dep)
        if not labels_mod.in_current_package(dep_label):
            fail("package_dep must be in current package: {}".format(
                pformat_ish(dep_label),
            ))

AnzuSharedLibraryMeta = provider(
    fields = [
        "package_dep_labels",
    ],
)

def _anzu_shared_library_meta_impl(ctx):
    return [
        AnzuSharedLibraryMeta(
            package_dep_labels = [
                dep.label
                for dep in ctx.attr.package_deps
            ],
        ),
    ]

_anzu_shared_library_meta = rule(
    implementation = _anzu_shared_library_meta_impl,
    attrs = {
        "package_deps": attr.label_list(),
    },
)

def _to_meta_name(name):
    return "_" + name + ".shared_library.meta"

def _add_shared_library_check(
        name,
        data,
        tags,
        package_deps,
        deps,
        verbose):
    # Instrument metadata.
    meta_name = _to_meta_name(name)
    _anzu_shared_library_meta(
        name = meta_name,
        package_deps = package_deps,
    )

    # Extract metadata.
    # TODO(eric.cousineau): How to attach this to existing cc rule?
    shared_deps_neighbor_meta = []
    for dep in deps:
        dep_label = labels_mod.to_label(dep)
        if not labels_mod.in_current_package(dep_label):
            continue
        if not _valid_shared_library_name(dep_label.name):
            continue
        shared_deps_neighbor_meta.append(_to_meta_name(dep_label.name))

    # Adds checko for dependencies and ensures we run our check rule when
    # building this target.
    analysis_name = "_" + name + ".shared_library.check"

    _anzu_shared_library_check(
        name = analysis_name,
        shared_library_name = name,
        static_deps = package_deps,
        shared_deps = deps,
        shared_deps_neighbor_meta = shared_deps_neighbor_meta,
        verbose = verbose,
        repository_name = native.repository_name(),
        package_name = native.package_name(),
    )

    # We need to ensure we trigger anaylsis somehow. Since the check
    # generates no data, we can add it to data. If/when it does
    # generate data,  it should be a separate rule.
    data = data + [":" + analysis_name]
    return data

def _assert_sane_shared_library_args(
        hdrs,
        srcs,
        data,
        package_deps,
        expert_mode):
    needs_expert = (
        package_deps == None or len(hdrs) > 0 or len(srcs) > 0
    )
    if needs_expert and not expert_mode:
        fail(
            "Please do not use `srcs`, `hdrs`, or `data`; instead use\n" +
            "`package_deps`. Use expert_mode = True if you think \n" +
            "you need them.",
        )

def _transform_shared_library_package_deps(
        name,
        hdrs,
        srcs,
        data,
        tags,
        deps,
        package_deps,
        package_deps_reuse_object_files,
        verbose):
    _assert_valid_package_deps(package_deps)

    new_hdrs, new_srcs, new_data = extract_package_cc_hdrs_srcs_data(
        name + "_extract",
        package_deps,
        reuse_object_files = package_deps_reuse_object_files,
    )
    hdrs = hdrs + new_hdrs
    srcs = srcs + new_srcs
    data = data + new_data

    if "anzu_shared_library_check_ignore" not in tags:
        data = _add_shared_library_check(
            name,
            data = data,
            tags = tags,
            package_deps = package_deps,
            deps = deps,
            verbose = verbose,
        )

    return hdrs, srcs, data

def _make_solib_name(name):
    pieces = (
        [native.repository_name()] +
        native.package_name().split("/") +
        [name]
    )
    solib = "lib{}.so.1".format(".".join(pieces))
    return solib

def anzu_cc_shared_library(
        name,
        hdrs = [],
        srcs = [],
        deps = [],
        data = [],
        tags = [],
        package_deps = None,
        package_deps_reuse_object_files = True,
        verbose = False,
        expert_mode = False,
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

    Arguments:
        srcs:
            C++ source files.
        hdrs:
            C++ header files.
        package_deps:
            If provdided, will scrape immediate (not transitive) headers and
            source files from these deps, which must reside in this package.
            Mutually exclusive to (srcs, hdrs).
            If supplied, will also add a check to ensure we have at least our
            expected dependencies and not any unallowed deps.
        package_deps_reuse_object_files:
            If True, will scrape object files from `package_deps`.
            If False, will recompile, which is useful for debugging
            dependencies at compile time rather than link time.
    """
    _assert_valid_shared_library_name(name)
    _assert_sane_shared_library_args(
        hdrs,
        srcs,
        data,
        package_deps,
        expert_mode,
    )

    if package_deps != None:
        hdrs, srcs, data = _transform_shared_library_package_deps(
            name = name,
            hdrs = hdrs,
            srcs = srcs,
            data = data,
            deps = deps,
            tags = tags,
            package_deps = package_deps,
            package_deps_reuse_object_files = package_deps_reuse_object_files,
            verbose = verbose,
        )

    solib = _make_solib_name(name)

    # Create main shared library.
    anzu_cc_binary(
        name = solib,
        srcs = srcs + hdrs,
        linkshared = 1,
        linkstatic = 1,
        data = data,
        deps = deps,
        tags = tags,
        **kwargs
    )

    # Expose shared library and headers for transitive dependencies.
    anzu_cc_library(
        name = name,
        hdrs = hdrs,
        srcs = [solib],
        deps = deps,
        tags = tags,
        **kwargs
    )

def anzu_third_party_cc_shared_library(
        name,
        hdrs = [],
        srcs = [],
        deps = [],
        data = [],
        tags = [],
        package_deps = None,
        package_deps_reuse_object_files = True,
        verbose = False,
        expert_mode = False,
        **kwargs):
    """Same as anzu_cc_shared_library, but explicitly for third-party
    repositories."""
    _assert_valid_shared_library_name(name)
    _assert_sane_shared_library_args(
        hdrs,
        srcs,
        data,
        package_deps,
        expert_mode,
    )

    if package_deps != None:
        hdrs, srcs, data = _transform_shared_library_package_deps(
            name = name,
            hdrs = hdrs,
            srcs = srcs,
            data = data,
            deps = deps,
            tags = tags,
            package_deps = package_deps,
            package_deps_reuse_object_files = package_deps_reuse_object_files,
            verbose = verbose,
        )

    solib = _make_solib_name(name)

    # Create main shared library.
    native.cc_binary(
        name = solib,
        srcs = srcs + hdrs,
        linkshared = 1,
        linkstatic = 1,
        deps = deps,
        data = data,
        tags = tags,
        **kwargs
    )

    # Expose shared library and headers for transitive dependencies.
    native.cc_library(
        name = name,
        hdrs = hdrs,
        srcs = [solib],
        deps = deps,
        tags = tags,
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
    _assert_valid_pybind_cc_deps(deps)
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

AnzuSharedLibraryInfo = provider(
    fields = [
        "direct_public_headers",
        # Transitive, but only in same package.
        "package_public_headers",
        # Transitive.
        "public_headers",
        # Map from header to (what is most likely) owning label.
        "public_headers_to_label",
        # Map from label to headers.
        "label_to_public_headers",
        # [ Indirect deps analysis ]
        # Map from label to dependency info (not including this target's).
        # Used
        "label_to_dep_info",
        # CcInfo for target.
        "cc_info",
    ],
)

def _in_our_workspaces(label):
    for workspace in _WORKSPACES_WITH_ANZU_SHARED_LIBRARY:
        if labels_mod.in_workspace(label, workspace):
            return True
    return False

def _should_collect_package_headers(target, dep):
    if not labels_mod.in_same_package(target.label, dep.label):
        return False
    return True

def _anzu_shared_library_aspect(target, ctx):
    deps = getattr(ctx.rule.attr, "deps", [])

    direct_public_headers = (
        target[CcInfo].compilation_context.direct_public_headers
    )
    dep_public_headers = []
    dep_package_public_headers = []
    public_headers_to_label = {
        hdr: target.label
        for hdr in direct_public_headers
    }
    label_to_public_headers = {}
    label_to_dep_info = {}
    for dep in deps:
        dep_info = dep[AnzuSharedLibraryInfo]
        dep_public_headers.append(dep_info.public_headers)
        public_headers_to_label.update(
            dep_info.public_headers_to_label,
        )
        label_to_public_headers.update(
            dep_info.label_to_public_headers,
        )
        if _should_collect_package_headers(target, dep):
            dep_package_public_headers.append(dep_info.package_public_headers)
        label_to_dep_info.update(dep_info.label_to_dep_info)
        dep_label = remap_label_for_anzu(dep.label)
        label_to_dep_info[dep_label] = dep_info
    public_headers = depset(
        direct_public_headers,
        transitive = dep_public_headers,
    )
    package_public_headers = depset(
        direct_public_headers,
        transitive = dep_package_public_headers,
    )
    target_label = remap_label_for_anzu(target.label)
    label_to_public_headers[target_label] = public_headers
    cc_info = target[CcInfo]

    return [
        AnzuSharedLibraryInfo(
            direct_public_headers = depset(direct_public_headers),
            package_public_headers = package_public_headers,
            public_headers = public_headers,
            public_headers_to_label = public_headers_to_label,
            label_to_public_headers = label_to_public_headers,
            label_to_dep_info = label_to_dep_info,
            cc_info = cc_info,
        ),
    ]

anzu_shared_library_aspect = aspect(
    implementation = _anzu_shared_library_aspect,
    attr_aspects = ["deps"],
)

def _get_transitive_shared_library_info_map(ctx, deps):
    label_to_info = {}

    # First accumulate dependency info.
    for dep in deps:
        info = dep[AnzuSharedLibraryInfo]
        label_to_info.update(info.label_to_dep_info)

    # Then accumulate direct info.
    for dep in deps:
        info = dep[AnzuSharedLibraryInfo]
        dep_label = remap_label_for_anzu(dep.label)
        label_to_info[dep_label] = info
    return label_to_info

def _extract_cc_object_files(ctx, dep_label, cc_info, mode):
    """
    Extracts C++ object files from `dep_label` and its corresponding `cc_info`.
    `mode` should be one of:
    - "transitive_package" - extract object files from this dependency and
       transitive dependencies that reside in the same current package.
    - "direct" - only extract object files from this dependency.
    """
    new_object_files, owners = extract_cc_object_files_ctx(ctx, cc_info)
    object_files = []
    object_file_to_label = {}
    for object_file, owner in zip(new_object_files, owners):
        if mode == "transitive_package":
            if not labels_mod.in_current_package(owner, ctx = ctx):
                continue
        elif mode == "direct":
            if owner != dep_label:
                continue
        else:
            fail("Invalid mode: " + mode)
        object_files.append(object_file)
        object_file_to_label[object_file] = owner
    return uniq(object_files), object_file_to_label

def _extract_cc_hdrs_single(ctx, shared_library_info, mode):
    """
    Extract C++ headers and a map from header to (what is most likely) the
    owning label.
    """

    # N.B. Take `shared_library_info` rather than `dep` so we can do our own
    # remapping for transitive deps and header extraction below.
    if mode == "transitive_package":
        hdrs = shared_library_info.package_public_headers
    elif mode == "direct":
        hdrs = shared_library_info.direct_public_headers
    elif mode == "transitive":
        hdrs = shared_library_info.public_headers
    else:
        fail("Invalid mode: " + mode)
    return hdrs, shared_library_info.public_headers_to_label

def _extract_cc_object_files_deps(
        ctx,
        dep_labels,
        cc_infos,
        mode):
    """
    Aggregates information from `_extract_cc_object_files_and_hdrs_single_dep`.
    """
    object_files = []
    object_files_to_dep_label = {}
    for dep_label, cc_info in zip(dep_labels, cc_infos):
        (
            new_object_files,
            new_object_files_to_dep_label,
        ) = _extract_cc_object_files(
            ctx,
            dep_label,
            cc_info,
            mode,
        )
        object_files_to_dep_label.update(new_object_files_to_dep_label)
        object_files += new_object_files
    object_files = depset(object_files)
    return object_files, object_files_to_dep_label

def _extract_cc_hdrs(ctx, shared_library_infos, mode):
    transitive_hdrs = []
    hdr_to_dep_label = {}
    for shared_library_info in shared_library_infos:
        (
            new_transitive_hdrs,
            new_hdr_to_dep_label,
        ) = _extract_cc_hdrs_single(
            ctx,
            shared_library_info = shared_library_info,
            mode = mode,
        )
        transitive_hdrs.append(new_transitive_hdrs)
        hdr_to_dep_label.update(new_hdr_to_dep_label)
    transitive_hdrs = depset(transitive = transitive_hdrs)
    return transitive_hdrs, hdr_to_dep_label

def _remap_to_shared_deps(
        ctx,
        dep_labels,
        cc_dep_allowlist_expanded,
        static_ignorable):
    """
    Remaps `dep_labels` to their equivalent shared dependency-friendly
    labels.
    """
    remapped = []
    maybe_invalid = []
    new_to_old = {}
    for dep_label in dep_labels.to_list():
        new_remapped, new_maybe_invalid = (
            _remap_to_shared_deps_from_single_label(
                ctx,
                dep_label,
                cc_dep_allowlist_expanded,
                static_ignorable,
            )
        )
        remapped += new_remapped
        maybe_invalid += new_maybe_invalid
        for new_label in new_remapped:
            if new_label not in new_to_old:
                new_to_old[new_label] = []
            new_to_old[new_label].append(dep_label)
    remapped = depset(remapped)
    maybe_invalid = depset(maybe_invalid)
    return remapped, maybe_invalid, new_to_old

def _try_static_remap_if_not_in_our_workspaces(
        ctx,
        static_dep_labels,
        cc_dep_allowlist_expanded):
    external = []
    external_remap = []
    ours = []
    for dep_label in static_dep_labels.to_list():
        if _in_our_workspaces(dep_label):
            ours.append(dep_label)
        else:
            new_remapped, _ = _remap_to_shared_deps_from_single_label(
                ctx,
                dep_label,
                cc_dep_allowlist_expanded,
                static_ignorable = True,
            )
            external.append(dep_label)
            external_remap += new_remapped
    external = depset(external)
    external_remap = depset(external_remap)
    ours = depset(ours)
    return external, external_remap, ours

def _anzu_shared_library_check_deps(ctx, label):
    verbose = ctx.attr.verbose
    if verbose:
        pprint_ish(dict(
            static_deps = ctx.attr.static_deps,
            shared_deps = ctx.attr.shared_deps,
        ))

    # User-friendly check should already happen; fail fast here.
    for dep in ctx.attr.static_deps:
        if not labels_mod.in_current_package(dep.label, ctx):
            fail("Must be in current package: " + pformat_ish(dep.label))

    static_dep_info = aggregate_extract_deps(
        [dep[ExtractDepsInfo] for dep in ctx.attr.static_deps],
    )

    shared_dep_info = aggregate_extract_deps(
        [dep[ExtractDepsInfo] for dep in ctx.attr.shared_deps],
    )
    all_dep_info = aggregate_extract_deps([static_dep_info, shared_dep_info])

    transitive_static_dep_labels = static_dep_info.dep_labels
    transitive_shared_dep_labels = shared_dep_info.dep_labels
    dep_label_map = all_dep_info.dep_label_map

    if verbose:
        pprint_ish(dict(all_dep_info = all_dep_info.dep_labels))

    # At present, we do not mind if we cannot expand all of these.
    # TODO(eric.cousineau): This does affect what deps are ultimately allowed.
    # Could solve by just supplying _CC_DEP_ALLOWLIST to the rule() to allow it
    # to always analyze the deps, assuming visilibity is not an issue.
    cc_dep_allowlist_expanded, _ = expand_label_deps(
        depset(_CC_DEP_ALLOWLIST),
        dep_label_map,
    )
    cc_dep_allowlist_expanded = cc_dep_allowlist_expanded.to_list()

    # Remap static to shared.
    static_deps_remap, static_deps_maybe_invalid, _ = (
        _remap_to_shared_deps(
            ctx,
            transitive_static_dep_labels,
            cc_dep_allowlist_expanded,
            # We can ignore static deps here.
            static_ignorable = True,
        )
    )
    static_deps_remap, static_deps_unexpanded = expand_label_deps(
        static_deps_remap,
        dep_label_map,
    )

    # Remap shared deps. We should not lose anything.
    shared_deps_remap, shared_deps_maybe_invalid, _ = (
        _remap_to_shared_deps(
            ctx,
            transitive_shared_dep_labels,
            cc_dep_allowlist_expanded,
            # We cannot ignore static deps, since these are our direct
            # dependencies.
            static_ignorable = False,
        )
    )
    shared_deps_remap, shared_deps_unexpanded = expand_label_deps(
        shared_deps_remap,
        dep_label_map,
    )

    if verbose:
        pprint_ish(dict(
            transitive_static_dep_labels = transitive_static_dep_labels,
            static_deps_remap = static_deps_remap,
            transitive_shared_dep_labels = transitive_shared_dep_labels,
            shared_deps_remap = shared_deps_remap,
        ))

    deps_unexpanded = combine_and_sort(
        [static_deps_unexpanded, shared_deps_unexpanded],
    )

    if len(deps_unexpanded) > 0:
        # N.B. We bail immediately, as remapping is used below and we may
        # end up accumulating confusing / noisy errors.
        fail(
            ("Unable to expand deps for {}.\nMost likely, you need " +
             "to add these to your shared_library(deps).")
                .format(pformat_ish(deps_unexpanded)),  # noqa
        )

    deps_maybe_invalid = combine_and_sort(
        [static_deps_maybe_invalid, shared_deps_maybe_invalid],
    )
    shared_deps_lost = set_diff(
        transitive_shared_dep_labels,
        shared_deps_remap,
    )

    if len(deps_maybe_invalid) == 0 and len(shared_deps_lost) > 0:
        # Perhaps a more advanced bug. Fail fast.
        fail(
            "Following deps for *shared_library() are lost on remap: " +
            pformat_ish(shared_deps_lost) +
            "Please consult / debug / change anzu_pybind.bzl.",
        )

    shared_deps_missing = set_diff(
        static_deps_remap,
        transitive_shared_dep_labels,
    )

    return (
        shared_deps_missing,
        deps_maybe_invalid,
        transitive_static_dep_labels,
        transitive_shared_dep_labels,
        cc_dep_allowlist_expanded,
    )

def _extract_cc_transitive_hdrs_via_labels(
        ctx,
        label_to_shared_library_info,
        dep_labels_lists):
    transitive_hdrs_lists = []
    hdr_to_label = {}
    for dep_labels in dep_labels_lists:
        shared_library_infos = [
            label_to_shared_library_info[dep_label]
            for dep_label in dep_labels.to_list()
        ]
        (transitive_hdrs, new_hdr_to_label) = _extract_cc_hdrs(
            ctx,
            shared_library_infos = shared_library_infos,
            mode = "transitive",
        )
        transitive_hdrs_lists.append(transitive_hdrs)
        hdr_to_label.update(new_hdr_to_label)
    return transitive_hdrs_lists, hdr_to_label

def _anzu_shared_library_check_compilation(
        ctx,
        transitive_static_dep_labels,
        transitive_shared_dep_labels,
        cc_dep_allowlist_expanded):
    verbose = ctx.attr.verbose

    label_to_shared_library_info = _get_transitive_shared_library_info_map(
        ctx,
        deps = ctx.attr.shared_deps + ctx.attr.static_deps,
    )

    direct_static_dep_labels = [dep.label for dep in ctx.attr.static_deps]
    direct_static_cc_infos = [dep[CcInfo] for dep in ctx.attr.static_deps]
    (
        needed_object_files,
        object_files_to_dep_label,
    ) = _extract_cc_object_files_deps(
        ctx,
        dep_labels = direct_static_dep_labels,
        cc_infos = direct_static_cc_infos,
        mode = "transitive_package",
    )

    actual_object_files, _ = _extract_cc_object_files_deps(
        ctx,
        dep_labels = direct_static_dep_labels,
        cc_infos = direct_static_cc_infos,
        mode = "direct",
    )

    shared_dep_neighbor_labels = []
    for meta in ctx.attr.shared_deps_neighbor_meta:
        meta_info = meta[AnzuSharedLibraryMeta]
        shared_dep_neighbor_labels += meta_info.package_dep_labels
    shared_dep_neighbor_cc_infos = [
        label_to_shared_library_info[dep_label].cc_info
        for dep_label in shared_dep_neighbor_labels
    ]
    shared_dep_neighbor_object_files, _ = _extract_cc_object_files_deps(
        ctx,
        dep_labels = shared_dep_neighbor_labels,
        cc_infos = shared_dep_neighbor_cc_infos,
        mode = "transitive_package",
    )

    needed_object_files = set_diff(
        needed_object_files,
        shared_dep_neighbor_object_files,
    )

    if verbose:
        pprint_ish(dict(
            needed_object_files = needed_object_files,
            actual_object_files = actual_object_files,
            shared_dep_neighbor_labels = shared_dep_neighbor_labels,
            shared_dep_neighbor_object_files = shared_dep_neighbor_object_files,  # noqa
        ))

    static_deps_external, static_deps_external_remap, static_deps_ours = (
        _try_static_remap_if_not_in_our_workspaces(
            ctx,
            transitive_static_dep_labels,
            cc_dep_allowlist_expanded,
        )
    )
    static_deps_external_novel = depset(set_diff(
        static_deps_external,
        static_deps_external_remap,
    ))
    static_deps = depset(
        [remap_label_for_anzu(dep.label) for dep in ctx.attr.static_deps],
    )
    shared_deps = depset(
        [remap_label_for_anzu(dep.label) for dep in ctx.attr.shared_deps],
    )

    if verbose:
        pprint_ish(dict(
            static_deps_external_novel = static_deps_external_novel,
            static_deps_external_remap = static_deps_external_remap,
            static_deps_ours = static_deps_ours,
            shared_deps = shared_deps,
        ))

    (
        (
            static_hdrs_external_novel,
            static_hdrs_external_remap,
            static_hdrs_ours,
            shared_transitive_hdrs,
        ),
        hdr_to_label,
    ) = _extract_cc_transitive_hdrs_via_labels(
        ctx,
        label_to_shared_library_info = label_to_shared_library_info,
        dep_labels_lists = [
            static_deps_external_novel,
            static_deps_external_remap,
            static_deps_ours,
            shared_deps,
        ],
    )

    direct_hdrs, _ = _extract_cc_hdrs(
        ctx,
        shared_library_infos = [
            label_to_shared_library_info[dep_label]
            for dep_label in static_deps.to_list()
        ],
        mode = "direct",
    )

    needed_transitive_hdrs = set_diff(
        depset(
            transitive = [static_hdrs_external_remap, static_hdrs_ours],
        ),
        static_hdrs_external_novel,
    )

    actual_transitive_hdrs = depset(
        transitive = [shared_transitive_hdrs, direct_hdrs],
    )

    if verbose:
        pprint_ish(dict(
            needed_transitive_hdrs = needed_transitive_hdrs,
            actual_transitive_hdrs = actual_transitive_hdrs,
        ))

    compilation_errors = []
    missing_deps = []
    transitive_missing_shared_deps = []
    transitive_missing_static_deps = []
    invalid_deps = []

    object_files_missing = set_diff(
        needed_object_files,
        actual_object_files,
    )
    if len(object_files_missing) > 0:
        compilation_errors.append(
            "Missing object files: {}".format(
                pformat_ish(object_files_missing),
            ),
        )
        missing_deps += [
            object_files_to_dep_label[object_file]
            for object_file in object_files_missing
        ]

    transitive_hdrs_missing = set_diff(
        needed_transitive_hdrs,
        actual_transitive_hdrs,
    )
    if len(transitive_hdrs_missing) > 0:
        compilation_errors.append(
            "Missing transitive headers: " +
            pformat_ish(
                to_str_list(transitive_hdrs_missing, truncate_at = 10),
            ),
        )
        hdr_deps = []
        for hdr in transitive_hdrs_missing:
            hdr_deps.append(hdr_to_label[hdr])
        new_missing_deps, new_invalid_deps, new_to_old = (
            _remap_to_shared_deps(
                ctx,
                depset(hdr_deps),
                cc_dep_allowlist_expanded,
                # TODO(eric): What is the right thing here?
                static_ignorable = False,
            )
        )
        invalid_deps += new_invalid_deps.to_list()
        transitive_shared_dep_labels = transitive_shared_dep_labels.to_list()
        for dep in new_missing_deps.to_list():
            if dep in transitive_shared_dep_labels:
                # This means a shared_dep needs to be updated.
                transitive_missing_shared_deps.append(dep)
                transitive_missing_static_deps += new_to_old[dep]
            else:
                missing_deps.append(dep)

    return (
        compilation_errors,
        missing_deps,
        invalid_deps,
        transitive_missing_shared_deps,
        transitive_missing_static_deps,
    )

def _anzu_shared_library_check_impl(ctx):
    # See docstring below.
    label = labels_mod.to_label(":" + ctx.attr.shared_library_name, ctx)

    (
        missing_deps,
        invalid_deps,
        transitive_static_dep_labels,
        transitive_shared_dep_labels,
        cc_dep_allowlist_expanded,
    ) = _anzu_shared_library_check_deps(ctx, label)

    (
        compilation_errors,
        new_missing_deps,
        new_invalid_deps,
        transitive_missing_shared_deps,
        transitive_missing_static_deps,
    ) = _anzu_shared_library_check_compilation(
        ctx,
        transitive_static_dep_labels,
        transitive_shared_dep_labels,
        cc_dep_allowlist_expanded,
    )

    missing_deps = uniq(missing_deps + new_missing_deps)
    invalid_deps = uniq(invalid_deps + new_invalid_deps)
    transitive_missing_shared_deps = uniq(transitive_missing_shared_deps)
    transitive_missing_static_deps = uniq(transitive_missing_static_deps)

    dep_errors = []

    if len(missing_deps) > 0:
        # TODO(eric.cousineau): Try reduce span on certain deps, e.g. @ros2.
        dep_errors.append(
            "The above errors can most likely be resolved by adding these " +
            "to anzu*_cc_shared_library(deps): " + pformat_ish(
                labels_mod.labels_to_str_list(missing_deps, ctx),
            ),
        )

    if len(invalid_deps) > 0:
        dep_errors.append(
            "These are presently invalid deps: " +
            pformat_ish(invalid_deps) + "\n" +
            # Should be novice-solvable bug.
            "Please consult anzu_pybind.bzl; change it only if you are " +
            "confident. Consider investigation using something like:\n" +
            "  bazel query 'somepath({}, {})'".format(
                pformat_ish(label),
                pformat_ish(invalid_deps[0]),
            ),
        )

    if len(transitive_missing_shared_deps) > 0:
        dep_errors.append(
            "The following libraries are already part of deps, but " +
            "components are still missing: " + pformat_ish(
                labels_mod.labels_to_str_list(
                    transitive_missing_shared_deps,
                    ctx,
                ),
            ) + "\n" +
            "In each respective package, make sure these deps are present:",
        )
        package_to_static = {}
        for dep_label in transitive_missing_static_deps:
            package = labels_mod.qualify_package(dep_label)
            if package not in package_to_static:
                package_to_static[package] = []
            package_to_static[package].append(dep_label)
        for package, dep_labels in package_to_static.items():
            dep_errors.append(
                indent(
                    "In {}: {}".format(
                        package,
                        pformat_ish([":" + x.name for x in dep_labels]),
                    ),
                    prefix = "    ",
                ),
            )

    errors = compilation_errors + dep_errors
    if errors:
        fail("For shared_library {}\n{}".format(
            label,
            "\n".join(errors),
        ))
    return []

_attrs = {
    "shared_library_name": attr.string(),
    "static_deps": attr.label_list(
        providers = [CcInfo],
        aspects = [extract_deps_aspect, anzu_shared_library_aspect],
    ),
    "shared_deps": attr.label_list(
        providers = [CcInfo],
        aspects = [extract_deps_aspect, anzu_shared_library_aspect],
    ),
    "shared_deps_neighbor_meta": attr.label_list(
        providers = [AnzuSharedLibraryMeta],
    ),
    "verbose": attr.bool(default = False),
}
_attrs.update(labels_mod.attrs)

"""
Checks the following for anzu_*_cc_shared_library(package_deps, deps):
- `deps` should cover the dependencies of `package_deps`.
- `package_deps` and `deps` should not have any invalid dependencies.
- The direct object files from `package_deps` should effectively be the same as
  the object files and provided by `package_deps`'s dependencies in the same
  package.
- The public headers of `package_deps` should be the same as public headers
  of `deps`.

Attributes:
    shared_library_name:
        Name of target, `anzu_*_cc_shared_library(name)`.
    static_deps:
        These are the static deps (via `anzu_*_cc_shared_library(package_deps)`
        that we wish to transform for use with shared libraries. These should
        only reside in the same exact package.
    shared_deps:
        These are the deps for the shared library (via
        `anzu_*_cc_shared_library(deps)`. They should not involve the object
        files that will be stracted from `static_deps`.
    verbose:
        Print additional info for debugging.

Notes:
- `static_deps` does not imply that all of the deps contained within are purely
  static; rather, it is meant for cc targets that are linked statically.
- While `static_deps` incorporates direct dependencies, `shared_deps` only
  incorporates external dependencies.
- This analysis *should* anything beyond first-order package depenencies. E.g.:
  - //b:shared_library depends on //a:shared_library
  - //b:shared_library effectively needs //a:bar
  - //a:shared_library only provides //a:foo
  * Note that this is only caught if the dependencies provide headers.

Negative Testing:
    Manual testing at the moment:
    - Go to `shared_library` target in
      `common/package.BUILD.bazel`,then in whatever order /
      combination...
    - Comment out ":a", see it complain.
    - Add `verbose = True` to the target.
    - With an error present, add `tags = ["anzu_shared_library_check_ignore"]`
      and see no error messages.

    For transitive deps:
    - Go to `//common:shared_library`, and remove ":b"
    - Try to build `//other:shared_library`
    - Restore
    - Go to `//other:shared_library` and remove '//common:shared_library` and
      try to build `//other:shared_library`.
"""

_anzu_shared_library_check = rule(
    implementation = _anzu_shared_library_check_impl,
    attrs = _attrs,
)
