workspace(name = "jupyter_integration")

# Copied from docs.

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

# Update the SHA and VERSION to the lastest version available here:
# https://github.com/bazelbuild/rules_python/releases.

SHA="84aec9e21cc56fbc7f1335035a71c850d1b9b5cc6ff497306f84cced9a769841"

VERSION="0.23.1"

http_archive(
    name = "rules_python",
    sha256 = SHA,
    strip_prefix = "rules_python-{}".format(VERSION),
    url = "https://github.com/bazelbuild/rules_python/releases/download/{}/rules_python-{}.tar.gz".format(VERSION,VERSION),
)

load("@rules_python//python:repositories.bzl", "py_repositories")

py_repositories()

load("@rules_python//python:pip.bzl", "pip_parse")

# Create a central repo that knows about the dependencies needed from
# requirements_lock.txt.
pip_parse(
   name = "pip_deps",
   requirements_lock = "//:requirements.txt",
)
# Load the starlark macro, which will define your dependencies.
load("@pip_deps//:requirements.bzl", "install_deps")
# Call it to define repos for your requirements.
install_deps()
