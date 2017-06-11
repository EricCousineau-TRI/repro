// Objective: Check if colwise() / rowwise() can be used for selection
// Followup: Implement simple row / column views to ease selection.
#include <iostream>
#include <vector>

#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

// NOTE: XprType should be `const T&` or `T&` if not using a view.
// Use perfect forwarding when able.
template <typename XprType>
class RowView {
 public:
  RowView(XprType xpr)
      : xpr_(xpr) {}

  int size() {
    return xpr_.rows();
  }

  void resize(int row_count) {
    cout << "resize: " << row_count << endl;
    xpr_.resize(row_count, xpr_.cols());
  }

  template <typename Other>
  void resizeOtherEigen(Other&& other, int row_count) const {
    other.resize(row_count, xpr_.cols());
  }
  template <typename Other>
  void resizeOther(Other&& other, int row_count) const {
    other.resize(row_count);
  }

  auto xpr() { return xpr_; }
  auto xpr() const { return xpr_; }

  auto operator[](int index) {
    return xpr_.row(index);
  }
  auto operator()(int index) {
    return operator[](index);
  }

  auto operator[](int index) const {
    return xpr_.row(index);
  }
  auto operator()(int index) const {
    return operator[](index);
  }

 private:
  // TODO(eric.cousineau): Add static_assertion.
  XprType xpr_;
};

template<typename XprType>
auto MakeRowView(XprType&& xpr) {
  return RowView<XprType>(std::forward<XprType>(xpr));
}

template <typename Src, typename Dest>
void select_indices(const vector<int> indices, Src&& src, Dest&& dest) {
  for (int i = 0; i < indices.size(); ++i) {
    cout << "select " << i << ": " << src[indices[i]] << endl;
    dest[i] = src[indices[i]];
  }
}

int main() {
  Matrix3d X;
  X <<
    1, 2, 3,
    4, 5, 6,
    7, 8, 9;


  RowVector3d X1; // = X.rowwise()[0];  // No dice.
  cout << X1 << endl;

  auto X_rows = MakeRowView(X);
  cout << X_rows[0] << endl;
  X_rows[2] *= 200;
  cout << X << endl;

  const auto& X_const = X;
  auto X_rows_const = MakeRowView(X_const);
  cout << X_rows_const[1] << endl;
  // X_rows_const[1] *= 10;

  MatrixXd X_sub(0, 3);
  auto X_sub_rows = MakeRowView(X_sub);
  X_rows.resizeOther(X_sub_rows, 2);
  select_indices({2, 0}, X_rows, X_sub_rows);
  cout << endl;
  cout << X_sub << endl;

  return 0;
}
