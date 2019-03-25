#!/usr/bin/env python3

import os
from os import unlink, mkdir, symlink
from os.path import abspath, basename, dirname, isabs, join
from subprocess import run, PIPE
from shutil import rmtree

# Configured by Bazel.
CONFIG = %{config}

PROPERTIES = [
    "INCLUDE_DIRECTORIES",
    "LINK_FLAGS",
    "LINK_DIRECTORIES",
    "LINK_LIBRARIES",
]


def template(src, dest, subs):
    with open(src) as f:
        text = f.read()
    for old, new in subs.items():
        text = text.replace(old, new)
    with open(dest, "w") as f:
        f.write(text)


def add_transitives_libs(libs, libdirs):
    # Adds transitive libs directly to `libs`.
    # (kinda dumb that Bazel / CMake doesn't figure this out, but meh...)
    ldd_env = dict(LD_LIBRARY_PATH=":".join(libdirs))
    check_libs = [lib for lib in libs if dirname(lib) in libdirs]
    lines = run(
        ["ldd"] + check_libs, env=ldd_env, check=True,
        stdout=PIPE, encoding="utf8").stdout.strip().split("\n")
    # Use libnames to permit shadowing (when useful).
    libnames = [basename(lib) for lib in libs]
    for line in lines:
        line = line.strip()
        if " => " not in line:
            continue
        _, _, lib, _ = line.split()
        libdir = dirname(lib)
        libname = basename(lib)
        if libdir in libdirs and libname not in libnames:
            print("Transitive: {}".format(lib))
            assert lib not in libs
            libs.append(lib)
            libnames.append(libname)


def libdir_order_preference_sort(xs):
    # Go through and bubble up each thing.
    xs = list(xs)
    out = []
    for pref in CONFIG["libdir_order_preference"]:
        prefix = abspath(pref) + "/"
        for x in list(xs):
            if x.startswith(prefix):
                out.append(x)
                xs.remove(x)
    # Add remaining.
    out += xs
    return out


def configure():
    template("CMakeLists.txt.in", "CMakeLists.txt", {
        "@NAME@": CONFIG["name"],
        "@PACKAGES@": " ".join(CONFIG["packages"]),
        "@PROPERTIES@": " ".join(PROPERTIES),
    })
    unlink("CMakeLists.txt.in")
    with open("empty.cc", "w") as f:
        f.write("")
    mkdir("build")
    cmake_flags = [
        "-D{}={}".format(k, v)
        for k, v in CONFIG["cache_entries"].items()]
    cmake_env = dict(CONFIG["env_vars"])
    cmake_env.update(
        PATH="/usr/local/bin:/usr/bin:/bin",
    )
    run(["cmake", ".."] + cmake_flags, check=True, cwd="build", env=cmake_env)
    unlink("empty.cc")
    props = dict()
    with open("build/props.txt") as f:
        for line in f:
            prop, value = line.strip().split("=")
            props[prop] = value and value.split(";") or []
    rmtree("build")
    mkdir("include")
    includes = []
    for include_path in props["INCLUDE_DIRECTORIES"]:
        assert isabs(include_path), include_path
        include = join("include", include_path.replace("/", "_"))
        symlink(include_path, include)
        includes.append(include)
    linkopts = list(props["LINK_FLAGS"])
    libdirs = list(props["LINK_DIRECTORIES"])
    libs = list(props["LINK_LIBRARIES"])
    libdirs_ignore = {"/usr/lib", "/usr/lib/x86_64-linux-gnu"}
    for lib in libs:
        if isabs(lib):
            libdir = dirname(lib)
            if libdir not in libdirs and libdir not in libdirs_ignore:
                libdirs.append(libdir)
    libdirs = libdir_order_preference_sort(libdirs)
    libs = libdir_order_preference_sort(libs)
    print("\n".join(libs))
    for libdir in libdirs:
        linkopts += ["-L" + libdir, "-Wl,-rpath " + libdir]
    add_transitives_libs(libs, set(libdirs))
    for lib in libs:
        linkopts += ["-l" + lib]
    template("BUILD.tpl", "BUILD.bazel", {
        "%{name}": repr(CONFIG["name"]),
        "%{includes}": repr(includes),
        "%{linkopts}": repr(linkopts),
        "%{deps}": repr(CONFIG["deps"]),
    })
    # TODO(eric.cousineau): How do we strip out unused shared libraries like
    # CMake does???
    unlink("BUILD.tpl")


def main():
    configure()


if __name__ == "__main__":
    main()
