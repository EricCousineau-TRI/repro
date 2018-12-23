#include <utility>

struct RefStruct {
  int& x;
};

int main() {
  int x{1};
  RefStruct s{x};
  RefStruct t{x};
  t = std::move(s);
  (void)t;

  return 0;
}
