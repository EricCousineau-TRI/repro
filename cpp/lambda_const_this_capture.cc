struct MyThing {
  int value{};

  auto MakeLambda() {
    return [const this]() {
      return value;
    };
  }
}

int main() {
  MyThing{.value = 10};
  auto lambda = MakeLambda();
  if (lambda() == 10) {
    return 0;
  } else {
    return 1;
  }
}
