#include <Eigen/Dense>
#include <fmt/format.h>

int main() {
    Eigen::MatrixX<int> A;
    Eigen::VectorX<int> b;
    fmt::print("A.size(): {}\n", A.size());
    fmt::print("b.size(): {}\n", b.size());
    return 0;
}
