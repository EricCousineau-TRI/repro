package(default_visibility = ["//visibility:public"])

# === CROSSTOOL rules ===

# The default toolchain selector for Drake.
#
# On OS X, we do not support any custom values for the --compiler flag. It
# must be unspecified.  The CROSSTOOL file maps this to the system default
# Clang via osx_cc_wrapper.sh.  Confusingly, Clang is installed on Mac as
# /usr/bin/gcc as well as /usr/bin/clang, and the wrapper script invokes the
# former.
#
# On Linux, we support --compiler strings "clang-3.9", "gcc-4.9", and "gcc-5",
# each of which does exactly what you would expect. "clang-3.9" is the default
# for convenience, because the Drake install instructions provide it on both
# Trusty and Xenial. No wrapper script is needed for any of these compilers, so
# the cc_toolchain rule does not provide one. Support for "gcc-6" is
# experimental without any guarantees. Note that it is not in our supported
# configurations yet. See
# http://drake.mit.edu/developers.html#supported-configurations for more
# information.
cc_toolchain_suite(
    name = "default-toolchain",
    toolchains = {
        #        "darwin|compiler": "cc_toolchain_apple",
        "k8|clang-3.9": "cc_toolchain_linux",
        "k8|gcc-4.9": "cc_toolchain_linux",
        "k8|gcc-5": "cc_toolchain_linux",
        "k8|gcc-6": "cc_toolchain_linux",
    },
)



cc_toolchain(
    name = "cc_toolchain_linux",
    all_files = ":linux_toolchain_deps",
    compiler_files = ":linux_toolchain_deps",
    cpu = "k8",
    dwp_files = ":linux_toolchain_deps",
    dynamic_runtime_libs = [":empty"],
    linker_files = ":linux_toolchain_deps",
    objcopy_files = ":linux_toolchain_deps",
    static_runtime_libs = [":empty"],
    strip_files = ":linux_toolchain_deps",
    supports_param_files = 0,
)

# This filegroup contains the files on which every toolchain component should
# always depend, regardless of platform. It is a useful place for the outputs
# of rules that check preconditions for the entire build.
filegroup(
    name = "universal_toolchain_deps",
    srcs = [], # ["ccache.txt"],  # ccache.txt is an output of :ccache_is_bad
    visibility = ["//visibility:private"],
)

filegroup(
    name = "linux_toolchain_deps",
    srcs = [
        ":universal_toolchain_deps",
    ],
    visibility = ["//visibility:private"],
)

config_setting(
    name = "gcc4.9-linux",
    values = {
        "compiler": "gcc-4.9",
        "cpu": "k8",
    },
)

config_setting(
    name = "linux",
    values = {"cpu": "k8"},
)

config_setting(
    name = "clang3.9-linux",
    values = {
        "compiler": "clang-3.9",
        "cpu": "k8",
    },
)
