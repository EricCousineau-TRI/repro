int counter();

extern "C" {
int wrapped_c() {
  return counter();
}
}
