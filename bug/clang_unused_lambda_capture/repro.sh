#!/bin/bash
set -u -o pipefail

CLANG=clang++-6.0
GCC=g++-5

test() {
    echo "test.cc"
    echo
    cat test.cc | sed 's#^#   #g'
    echo
    args="-std=c++14 -Wall -Werror ./test.cc -o ./test"
    (set -x; ${CLANG} ${args}) && (set -x; ./test)
    echo
    (set -x; ${GCC} ${args}) && (set -x; ./test)
    echo "---"
    echo
    echo
}

(set -x; ${CLANG} --version)
echo
(set -x; ${GCC} --version)
echo

echo '[ 0: static storage, with capture: fails clang ]'
cat > test.cc <<EOF
#include <iostream>
const char top_doc[] = "Works";
int main() {
  auto& doc = top_doc;
  [&doc]() { std::cout << doc << std::endl; }();
  return 0;
}
EOF
test

echo '[ 1: static storage, w/o capture: fails gcc ]'
cat > test.cc <<EOF
#include <iostream>
const char top_doc[] = "Works";
int main() {
  auto& doc = top_doc;
  []() { std::cout << doc << std::endl; }();
  return 0;
}
EOF
test

echo '[ 2: automatic storage: works ]'
cat > test.cc <<EOF
#include <iostream>
int main() {
  const char top_doc[] = "Works";
  auto& doc = top_doc;
  [&doc]() { std::cout << doc << std::endl; }();
  return 0;
}
EOF
test

echo '[ 3: auto& infer constexpr, with capture: fails clang ]'
cat > test.cc <<EOF
#include <iostream>
constexpr char top_doc[] = "Works";
int main() {
  auto& doc = top_doc;
  [&doc]() { std::cout << doc << std::endl; }();
  return 0;
}
EOF
test

echo '[ 4: auto& infer constexpr, without capture: fails gcc ]'
cat > test.cc <<EOF
#include <iostream>
constexpr char top_doc[] = "Works";
int main() {
  auto& doc = top_doc;
  []() { std::cout << doc << std::endl; }();
  return 0;
}
EOF
test

echo '[ 5: constexpr auto&: works ]'
cat > test.cc <<EOF
#include <iostream>
constexpr char top_doc[] = "Works";
int main() {
  constexpr auto& doc = top_doc;
  []() { std::cout << doc << std::endl; }();
  return 0;
}
EOF
test
