#!/usr/bin/env python3

"""
Example of doing cmdline script-y things in Python (rather than a bash script).

Derived from:
https://github.com/RobotLocomotion/drake/blob/7de24898/tmp/benchmark/generate_benchmark_from_master.py
"""

from contextlib import closing
import os
from os.path import abspath, dirname, isfile
from subprocess import PIPE, run
import sys
from textwrap import indent

import numpy as np
import pyassimp
import yaml

from process_util import CapturedProcess, bind_print_prefixed


class UserError(RuntimeError):
    pass


def eprint(s):
    print(s, file=sys.stderr)


def shell(cmd, check=True):
    """Executes a shell command."""
    eprint(f"+ {cmd}")
    return run(cmd, shell=True, check=check)


def subshell(cmd, check=True, stderr=None, strip=True):
    """Executs a subshell in a capture."""
    eprint(f"+ $({cmd})")
    result = run(cmd, shell=True, stdout=PIPE, stderr=stderr, encoding="utf8")
    if result.returncode != 0 and check:
        if stderr == PIPE:
            eprint(result.stderr)
        eprint(result.stdout)
        raise UserError(f"Exit code {result.returncode}: {cmd}")
    out = result.stdout
    if strip:
        out = out.strip()
    return out


def cd(p):
    eprint(f"+ cd {p}")
    os.chdir(p)


def parent_dir(p, *, count):
    for _ in range(count):
        p = dirname(p)
    return p


def load_mesh(mesh_file):
    assert isfile(mesh_file), mesh_file
    scene = pyassimp.load(mesh_file)
    assert scene is not None, mesh_file
    return scene


def get_mesh_extent(scene, mesh_file):
    # Return geometric center and size.

    # def check_identity(node):
    #     np.testing.assert_equal(
    #         node.transformation,
    #         np.eye(4),
    #         err_msg=mesh_file,
    #     )
    #     for child in node.children:
    #         check_identity(child)

    # assert len(scene.meshes) > 0, mesh_file
    # # Meshes should not have transforms.
    # check_identity(scene.rootnode)

    v_list = []
    for mesh in scene.meshes:
        v_list.append(mesh.vertices)
    v = np.vstack(v_list)
    lb = np.min(v, axis=0)
    ub = np.max(v, axis=0)
    size = ub - lb
    center = (ub + lb) / 2
    return np.array([center, size])


def convert_file_to_obj(mesh_file, suffix):
    assert mesh_file.endswith(suffix), mesh_file
    obj_file = mesh_file[:-len(suffix)] + ".obj"
    print(f"Convert Mesh: {mesh_file} -> {obj_file}")
    if isfile(obj_file):
        return
    scene = load_mesh(mesh_file)

    #Workaround for issue #849
    scene.mRootNode.contents.mTransformation = \
        pyassimp.structs.Matrix4x4(1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1)

    pyassimp.export(scene, obj_file, file_type="obj")

    # TODO (marcoag) skip sanity check for now
    # find a way to do one
    # Sanity check.
#    extent = get_mesh_extent(scene, mesh_file)
#    scene_obj = load_mesh(obj_file)
#    extent_obj = get_mesh_extent(scene_obj, mesh_file)
#    np.testing.assert_equal(
#        extent, extent_obj,
#        err_msg=repr((mesh_file, obj_file)),
#    )


def replace_text_to_obj(content, suffix):
    # Not robust, but meh.
    return content.replace(suffix, ".obj")


def find_mesh_files(d, suffix):
    files = subshell(f"find meshes -name '*{suffix}'").strip().split()
    files.sort()
    return files


FLAVORS = [
    "ur3",
    "ur3e",
    "ur5",
    "ur5e",
]


def main():
    source_tree = parent_dir(abspath(__file__), count=1)
    cd(source_tree)

    print(pyassimp.__file__)
    print(pyassimp.core._assimp_lib.dll)

    if "ROS_DISTRO" not in os.environ:
        raise UserError("Please run under `./ros_setup.bash`, or whatevs")

    cd("repos/universal_robot")
    # Use URI that is unlikely to be used.
    os.environ["ROS_MASTER_URI"] = "http://localhost:11321"
    os.environ[
        "ROS_PACKAGE_PATH"
    ] = f"{os.getcwd()}:{os.environ['ROS_PACKAGE_PATH']}"

    cd("ur_description")

    print()
    print(f"[ Convert Meshes for Drake :( ]")
    for dae_file in find_mesh_files("meshes", ".dae"):
        convert_file_to_obj(dae_file, ".dae")
    for stl_file in find_mesh_files("meshes", ".stl"):
        convert_file_to_obj(stl_file, ".stl")

    urdf_files = []
    # Start a roscore, 'cause blech.
    roscore = CapturedProcess(
        ["roscore", "-p", "11321"],
        on_new_text=bind_print_prefixed("[roscore] "),
    )
    with closing(roscore):
        # Blech.
        while "started core service" not in roscore.output.get_text():
            assert roscore.poll() is None

        for flavor in FLAVORS:
            shell(f"roslaunch ur_description load_{flavor}.launch")
            urdf_file = f"urdf/{flavor}.urdf"
            output = subshell(f"rosparam get /robot_description")
            # Blech :(
            content = yaml.load(output)
            content = replace_text_to_obj(content, ".stl")
            content = replace_text_to_obj(content, ".dae")
            with open(urdf_file, "w") as f:
                f.write(content)
            urdf_files.append(urdf_file)

        print("\n\n")
        print("Generated URDF files:")
        print(indent("\n".join(urdf_files), "  "))


if __name__ == "__main__":
    try:
        main()
        print()
        print("[ Done ]")
    except UserError as e:
        eprint(e)
        sys.exit(1)
