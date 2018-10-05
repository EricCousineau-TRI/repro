#!/bin/bash
set -u -o pipefail

test() {
    echo "test.cc"
    echo
    cat test.cc | sed 's#^#   #g'
    echo
    args="-std=c++14 -Wall -Werror ./test.cc -o ./test"
    (
        set -x
        clang++-6.0 ${args} && ./test
        g++-5 ${args} && ./test
    )
    echo
    echo
}

echo '[ non-auto storage: fails ]'
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

echo '[ auto storage: works ]'
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

echo '[ auto&: fails ]'
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

echo '[ constexpr auto&: works ]'
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
