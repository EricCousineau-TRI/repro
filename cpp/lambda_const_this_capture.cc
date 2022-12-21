struct MyThing {
  int value{};

  auto MakeLambda() {
    // TODO(eric.cousineau): Better method for this?
    const auto& self = *this;
    return [&self]() {
      // self.value = 3;  // should fail compilation
      return self.value;
    };
  }
};

int main() {
  MyThing my_thing{.value = 10};
  auto lambda = my_thing.MakeLambda();
  if (lambda() == 10) {
    return 0;
  } else {
    return 1;
  }
}
