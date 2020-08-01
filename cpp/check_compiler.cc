#include <iostream>

#include "clang/AST/Stmt.h"

int main() {
  const int cplusplus = __cplusplus;
  const int clang_major = __clang_major__;
  const int clang_minor = __clang_minor__;

  std::cout
    << "cplusplus: " << cplusplus << std::endl
    << "clang_major: " << clang_major << std::endl
    << "clang_minor: " << clang_minor << std::endl
    << "arbitrary clang symbol RTTI: "
        << typeid(clang::Stmt).name() << std::endl;
  return 0;
}

/*
$ apt install clang-9 llvm-9-dev
$ clang++-9 \
    -I/usr/lib/llvm-9/include \
    -L/usr/lib/llvm-9/lib -lclang-cpp -lLLVM-9.0.0 \
    --std=c++17 ./check_compiler.cc -o /tmp/bin \
    && /tmp/bin
cplusplus: 201703
clang_major: 9
clang_minor: 0
arbitrary clang symbol RTTI: N5clang4StmtE
*/
