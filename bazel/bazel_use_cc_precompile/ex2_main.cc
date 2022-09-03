#include <stdio.h>
#include <dlfcn.h>
#include <assert.h>

#include <filesystem>
#include <string>

#include "tools/cpp/runfiles/runfiles.h"
using bazel::tools::cpp::runfiles::Runfiles;

typedef int (*func_t)();
func_t wrapped_b = nullptr;
func_t wrapped_c = nullptr;

func_t load_func(
    const std::string& path,
    const std::string& func_name) {
  void* lib = dlopen(path.c_str(), RTLD_LAZY);
  assert(lib != nullptr);
  func_t func = (func_t)dlsym(lib, func_name.c_str());
  assert(dlerror() == nullptr);
  return func;
}

void init() {
  const std::string& argv0 = std::filesystem::read_symlink({
        "/proc/self/exe"}).string();
  std::unique_ptr<Runfiles> runfiles(Runfiles::Create(argv0));
  assert(runfiles != nullptr);
  wrapped_b = load_func(
      runfiles->Rlocation("workspace/libex2_b.so.1"), "wrapped_b");
  wrapped_c = load_func(
    runfiles->Rlocation("workspace/libex2_c.so.1"), "wrapped_c");
}

void call() {
  printf("wrapped_b: %d\n", (*wrapped_b)());
  printf("wrapped_c: %d\n", (*wrapped_c)());
  printf("\n");
}

int main() {
  init();
  call();
  call();
  return 0;
}
