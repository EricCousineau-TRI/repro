#!/bin/bash
set -eux

# Build with Bazel.
bazel build //:libtorch_test

torch_dir=bazel-libtorch_example/external/libtorch

# CMake
# # Doesn't work if CUDA isn't installed? Why?
# mkdir -p build_cmake && cd build_cmake
# cmake .. -DCMAKE_PREFIX_PATH=${torch_dir}
# make

# Direct build.
mkdir -p build_direct
# use ./print_libs.py to produce libraries.
g++ \
    libtorch_test.cc \
    -o ./build_direct/libtorch_test \
    --std=c++17 \
    -I${torch_dir}/include \
    -I${torch_dir}/include/torch/csrc \
    -I${torch_dir}/include/torch/csrc/api/include \
    -L${torch_dir}/lib \
    -Wl,-rpath,'$ORIGIN'/../${torch_dir}/lib \
    -l:libbackend_with_compiler.so \
    -l:libc10.so \
    -l:libc10_cuda.so \
    -l:libc10d_cuda_test.so \
    -l:libcaffe2_detectron_ops_gpu.so \
    -l:libcaffe2_module_test_dynamic.so \
    -l:libcaffe2_nvrtc.so \
    -l:libcaffe2_observers.so \
    -l:libcudart-a7b20f20.so.11.0 \
    -l:libgomp-52f2fd74.so.1 \
    -l:libjitbackend_test.so \
    -l:libnnapi_backend.so \
    -l:libnvToolsExt-24de1d56.so.1 \
    -l:libnvrtc-1ea278b5.so.11.2 \
    -l:libnvrtc-builtins-4730a239.so.11.3 \
    -l:libshm.so \
    -l:libtorch.so \
    -l:libtorch_cpu.so \
    -l:libtorch_cuda.so \
    -l:libtorch_cuda_cpp.so \
    -l:libtorch_cuda_cu.so \
    -l:libtorch_global_deps.so \
    -l:libtorch_python.so \
    -l:libtorchbind_test.so

# Inspect results.

scrub() {
    sed -E -e "s#${PWD}#{proj}#g" -e "s#0x[0-9a-f]+#0x{hex}#g"
}

(
    ! bazel-bin/libtorch_test
    echo
    ldd bazel-libtorch_example/external/libtorch/lib/{*.so,*.so.*}
    echo
    ldd bazel-bin/libtorch_test
    echo
    # Fails about CUDA?
    ! build_direct/libtorch_test
    echo
    ldd build_direct/libtorch_test
) 2>&1 | scrub | tee repro.output.txt
