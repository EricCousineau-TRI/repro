load(":vtk.bzl", "vtk_includes", "vtk_lib")

cc_binary(
    name = "particle_reader",
    srcs = [
        "ParticleReader.cxx",
    ],
    includes = vtk_includes(),
    deps = [
        vtk_lib("vtkHybrid"),
    ],
)
