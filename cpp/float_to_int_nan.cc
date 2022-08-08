#include <iostream>

using namespace std;

int main() {
  double oops_nan_f = std::numeric_limits<double>::quiet_NaN();
  int32_t oops_nan_int = static_cast<int32_t>(oops_nan_f);
  cout
      << oops_nan_f << endl
      << oops_nan_int << endl;

  return 0;
}

/*
Output:

nan
0
*/
