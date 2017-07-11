#include <functional>
#include <iostream>
#include <map>
#include <string>

using namespace std;

namespace std {

template <typename A, typename B>
bool operator<(const reference_wrapper<A>& a, const reference_wrapper<B>& b) {
  return a.get() < b.get();
}

}

// TODO:
// https://stackoverflow.com/a/10717235/7829525

template <typename K, typename V>
using cref_map = map<reference_wrapper<const K>, reference_wrapper<const V>>;

int main() {
  string a_k = "a";
  int a_v = 1;

  string b_k = "a";
  int b_v = 2;

  // The key should NOT be changed.
  cref_map<string, int> x;
  // x[a_k] = a_v;  // Will try to create an empty type.
  // x.insert(cref_map<string, int>::value_type(a_k, a_v));
  x.insert({a_k, a_v});
  x.insert({b_k, b_v});
  // x[b_k] = b_v;


  cout << x.at(a_k) << endl;
  a_v *= 10;
  cout << x.at(a_k) << endl;

  return 0;
}
