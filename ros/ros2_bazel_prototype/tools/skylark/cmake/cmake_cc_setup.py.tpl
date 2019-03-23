#!/usr/bin/env python3

import os
from os import unlink, mkdir, symlink
from os.path import dirname, isabs, join
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
    # This is dumb, but we gotta find 'em all.
    ldd_env = dict(LD_LIBRARY_PATH=":".join(libdirs))
    check_libs = [lib for lib in libs if dirname(lib) in libdirs]
    lines = run(
        ["ldd"] + check_libs, env=ldd_env, check=True,
        stdout=PIPE, encoding="utf8").stdout.strip().split("\n")
    for line in lines:
        line = line.strip()
        if " => " not in line:
            continue
        _, _, lib, _ = line.split()
        libdir = dirname(lib)
        if libdir in libdirs and lib not in libs:
            print("Hidden: {}".format(lib))
            libs.append(lib)


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
        HOME=os.environ["HOME"],
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
    libdirs = set(props["LINK_DIRECTORIES"])
    libs = list(props["LINK_LIBRARIES"])
    for lib in libs:
        assert isabs(lib), lib
        libdirs.add(dirname(lib))
    libdirs -= {"/usr/lib", "/usr/lib/x86_64-linux-gnu"}
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
    unlink("BUILD.tpl")


def main():
    configure()


if __name__ == "__main__":
    main()
