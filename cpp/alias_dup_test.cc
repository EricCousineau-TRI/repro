namespace A {

int value = 1;

};

namespace B = A;
namespace B = A;

int main() {
  if (B::value == 1) {
    return 0;
  } else {
    return 1;
  }
}
