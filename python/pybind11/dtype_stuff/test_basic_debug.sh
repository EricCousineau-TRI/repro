(
    set -x -e -u
    target=//python/pybind11/dtype_stuff:test_basic
    target_bin=$(echo ${target} | sed -e 's#//##' -e 's#:#/#')
    bazel run -c dbg ${target} -j 8 || :
    workspace=$(bazel info workspace)
    name=$(basename ${workspace})
    target_bin_path=${workspace}/bazel-bin/${target_bin}
    source_dir=${workspace}/bazel-${name}
    source /tmp/env.sh
    cd ${target_bin_path}.runfiles/${name}
    valgrind --tool=memcheck --leak-check=full ${target_bin_path}
    # gdb --directory ${source_dir} ${target_bin_path}
)
