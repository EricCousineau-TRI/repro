int good_thing() {
  int x = 10;
  // Good instruction.
  x += 1;
  return x;
}

void bad_thing() {
  int* px = 0;
  // Purposeful segfault.
  *px = 0xbadf00d;
}

int main() {
  good_thing();
  bad_thing();
  return 0;
}
