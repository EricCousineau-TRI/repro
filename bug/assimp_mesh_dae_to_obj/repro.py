from os.path import abspath, dirname, isfile
from textwrap import indent

import numpy as np
import pyassimp


def load_mesh(mesh_file):
    assert isfile(mesh_file), mesh_file
    scene = pyassimp.load(mesh_file)
    assert scene is not None, mesh_file
    return scene


def get_mesh_vertices(scene):

    def print_transforms(node, prefix=""):
        print(indent(str(node.transformation), prefix))
        for child in node.children:
            print_transforms(child, prefix=f"{prefix}  ")

    print_transforms(scene.rootnode)

    v_list = []
    for mesh in scene.meshes:
        v_list.append(mesh.vertices)
    v = np.vstack(v_list)
    assert v.size > 0
    return v


def convert_mesh(old_mesh_file, new_mesh_file, file_type):
    print(f"Original: {old_mesh_file}")
    scene_old = load_mesh(old_mesh_file)
    v_old = get_mesh_vertices(scene_old)

    pyassimp.export(scene_old, new_mesh_file, file_type=file_type)
    print(f"Converted: {new_mesh_file}")
    scene_new = load_mesh(new_mesh_file)
    v_new = get_mesh_vertices(scene_new)

    # Sanity check.
    np.testing.assert_allclose(v_old, v_new, atol=1e-8, rtol=0)


def print_assimp_info():
    print(f"pyassimp: {pyassimp.__file__}")
    print(f"  dll: {pyassimp.core._assimp_lib.dll}")
    print()


def main():
    print_assimp_info()
    convert_mesh("base.dae", "base.obj", "obj")
    print("[ Done ]")


assert __name__ == "__main__"
main()
