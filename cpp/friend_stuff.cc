
class Yar;

template <typename U, typename Token>
struct stuff {
  using T = typename U::PrivateT;
};

class Yar {
 private:
  friend struct stuff<Yar, int>;
  using PrivateT = double;
};

int main() {
  stuff<Yar, int>::T x{};
  stuff<Yar, void>::T x{};
  return 0;
}
