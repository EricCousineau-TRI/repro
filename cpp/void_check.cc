int func() {
  return 0;
}

void stuff() {}

// No dice.
int main() {
  return func(stuff());
}
