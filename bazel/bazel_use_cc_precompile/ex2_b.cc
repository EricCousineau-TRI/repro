int counter();

extern "C" {
int wrapped_b() {
  return counter();
}
}
