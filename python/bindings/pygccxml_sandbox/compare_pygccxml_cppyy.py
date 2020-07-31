"""
Only tested on Ubuntu 18.04. Change to this directory, then run:

    source ./setup.sh
    python ./compare_pygccxml_cppyy.py
"""

import multiprocessing as mp
import time

import numpy as np
import tqdm


def run_isolated(f, *args, **kwargs):
    # N.B. We don't really need this for pygccxml, but cppyy needs it 'cause I
    # (Eric) don't know how to clear out clang's memory and avoid duplicate
    # definition stuff.
    queue = mp.Queue(maxsize=1)

    def wrap():
        queue.put(f(*args, **kwargs))

    proc = mp.Process(target=wrap)
    proc.start()
    proc.join()
    return queue.get()


def decorate_isolated(f):

    def wrapper(*args, **kwargs):
        return run_isolated(f, *args, **kwargs)

    return wrapper


code = r"""
#include <vector>

#include <Eigen/Dense>

namespace ns {

template <typename T, typename U = int>
class ExampleClass {
public:
    std::vector<T> make_std_vector() const;
    Eigen::Matrix<U, 3, 3> make_matrix3();
};

// Analyze concrete instantiations of the given class.
extern template class ExampleClass<int>;
extern template class ExampleClass<double, double>;

}  // namespace ns
"""


@decorate_isolated
def try_pygccxml():
    from pygccxml import declarations
    from pygccxml import utils
    from pygccxml import parser

    # Find out the c++ parser. This should resolve to the castxml
    # version installed in Docker.
    generator_path, generator_name = utils.find_xml_generator()

    # Configure the xml generator
    config = parser.xml_generator_configuration_t(
        xml_generator_path=generator_path,
        xml_generator=generator_name,
        include_paths=["/usr/include/eigen3"],
        # TODO(eric.cousineau): Why is `compiler_path` necessary?
        compiler_path=generator_path,
        start_with_declarations=["ns"],
    )

    t_start = time.time()
    (global_ns,) = parser.parse_string(code, config)
    dt = time.time() - t_start
    return dt


@decorate_isolated
def try_clang_cindex():
    from clang.cindex import Index, Config, TranslationUnit
    Config.set_library_file(f"/usr/lib/llvm-9/lib/libclang.so")

    filename = "/tmp/clang_cindex_tmp_src.cc"
    with open(filename, "w") as f:
        f.write(code)

    index = Index.create()
    t_start = time.time()
    tu = TranslationUnit.from_source(
        filename=filename,
        index=index,
        args=[
            f"-I/usr/include/eigen3",
        ],
    )
    dt = time.time() - t_start
    return dt


@decorate_isolated
def try_cppyy():
    import cppyy
    cppyy.add_include_path("/usr/include/eigen3")

    def access(cls):
        cls.make_std_vector
        cls.make_matrix3

    t_start = time.time()

    cppyy.cppdef(code)    
    # Access some members to try and make it even for cppyy's lazy
    # instantiation.
    ns = cppyy.gbl.ns
    ns.ExampleClass[int]
    ns.ExampleClass[float, float]

    dt = time.time() - t_start
    return dt


def benchmark(f):
    ts = []
    for _ in tqdm.tqdm(range(10)):
        ts.append(f())
    print(f"Mean Time: {np.mean(ts)}")


def main():
    print("clang.cindex")  # 0.66s
    benchmark(try_clang_cindex)

    print("cppyy")  # 0.60s
    benchmark(try_cppyy)

    print("pygccxml")  # 0.98s
    benchmark(try_pygccxml)


if __name__ == "__main__":
    main()
