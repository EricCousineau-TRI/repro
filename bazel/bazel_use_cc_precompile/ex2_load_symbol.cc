#include "ex2_load_symbol.h"

#include <dlfcn.h>

#include <cassert>
#include <cstdio>
#include <filesystem>
#include <string>

#include "tools/cpp/runfiles/runfiles.h"
using bazel::tools::cpp::runfiles::Runfiles;

namespace {
std::unique_ptr<Runfiles> runfiles;
}  // namespace

std::string Rlocation(const std::string& respath) {
  if (runfiles == nullptr) {
    const std::string& argv0 = std::filesystem::read_symlink({
        "/proc/self/exe"}).string();
    runfiles.reset(Runfiles::Create(argv0));
    assert(runfiles != nullptr);
  }
  return runfiles->Rlocation(respath);
}

func_t LoadSymbol(const std::string& path, const std::string& func_name) {
  void* lib = dlopen(path.c_str(), RTLD_LAZY);
  assert(lib != nullptr);
  func_t func = (func_t)dlsym(lib, func_name.c_str());

  const char* error = dlerror();
  if (error) {
    printf("%s\n", error);
    exit(1);
  }
  return func;
}
