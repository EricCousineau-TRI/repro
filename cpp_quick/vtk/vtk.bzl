
VTK_ROOT = "/home/eacousineau/proj/tri/proj/dart_impl/install/vtk"
VTK_VERSION = "vtk-5.10"
VTK_INCLUDE = VTK_ROOT + "/include/" + VTK_VERSION
VTK_LIBDIR = VTK_ROOT + "/lib/" + VTK_VERSION

def vtk_includes():
    return [VTK_INCLUDE]

def vtk_lib(name):
    return "{}/lib{}.a".format(VTK_LIBDIR, name)
