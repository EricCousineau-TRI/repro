#!/bin/bash
set -u -o pipefail

CLANG=clang++-6.0
GCC=g++-7

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

echo '[ 0: constexpr auto& + non-generic lambda: works ]'
cat > test.cc <<EOF
#include <iostream>
constexpr char top_doc[] = "Works";
int main() {
  constexpr auto& doc = top_doc;
  [](int) { std::cout << doc << std::endl; }(1);
  return 0;
}
EOF
test

echo '[ 1: constexpr auto& + generic lambda: fails in gcc ]'
cat > test.cc <<EOF
#include <iostream>
constexpr char top_doc[] = "Works";
int main() {
  constexpr auto& doc = top_doc;
  [](auto) { std::cout << doc << std::endl; }(1);
  return 0;
}
EOF
test

echo '[ 2: constexpr + non-generic lambda: fails in clang ]'
cat > test.cc <<EOF
#include <iostream>
int main() {
  constexpr char doc[] = "Works";
  [](int) { std::cout << doc << std::endl; }(1);
  return 0;
}
EOF
test

echo '[ 3: constexpr + generic lambda: fails in clang ]'
cat > test.cc <<EOF
#include <iostream>
int main() {
  constexpr char doc[] = "Works";
  [](auto) { std::cout << doc << std::endl; }(1);
  return 0;
}
EOF
test
