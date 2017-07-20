// Objective: Check if colwise() / rowwise() can be used for selection
// Followup: Implement simple row / column views to ease selection.
#include <iostream>
#include <vector>

#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

// NOTE: XprType should be `const T&` or `T&` if not using a view.
// Use perfect forwarding when able.
// This will ONLY work if .row(int) returns a reference object that does
// not need to be forwarded.
template <typename XprType>
class RowView {
 public:
  RowView(XprType xpr)
      : xpr_(xpr) {}

  int size() const {
    return xpr_.rows();
  }

  void resize(int row_count) {
    xpr_.resize(row_count, xpr_.cols());
  }

  template <typename Other>
  void resizeLike(const RowView<Other>& other, int row_count = -1) const {
    int cols = other.xpr().cols();
    int rows = row_count == -1 ? other.xpr().rows() : row_count;
    xpr_.resize(rows, cols);
  }

  auto xpr() { return xpr_; }
  auto xpr() const { return xpr_; }

  auto segment(int index, int row_count) {
    return xpr_.middleRows(index, row_count);
  }

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


template <typename XprType>
class ColView {
 public:
  ColView(XprType xpr)
      : xpr_(xpr) {}

  int size() const {
    return xpr_.cols();
  }

  void resize(int col_count) {
    xpr_.resize(col_count, xpr_.cols());
  }

  template <typename Other>
  void resizeLike(const ColView<Other>& other, int col_count = -1) const {
    int cols = col_count == -1 ? other.xpr().cols() : col_count;
    int rows = other.xpr().rows();
    xpr_.resize(rows, cols);
  }

  auto xpr() { return xpr_; }
  auto xpr() const { return xpr_; }

  auto segment(int index, int col_count) {
    return xpr_.middleCols(index, col_count);
  }

  auto operator[](int index) {
    return xpr_.col(index);
  }
  auto operator()(int index) {
    return operator[](index);
  }

  auto operator[](int index) const {
    return xpr_.col(index);
  }
  auto operator()(int index) const {
    return operator[](index);
  }

 private:
  // TODO(eric.cousineau): Add static_assertion.
  XprType xpr_;
};

template<typename XprType>
auto MakeColView(XprType&& xpr) {
  return ColView<XprType>(std::forward<XprType>(xpr));
}



template <typename Src, typename Dest>
void select_indices(const vector<int> indices, Src&& src, Dest&& dest) {
  for (int i = 0; i < indices.size(); ++i) {
    dest[i] = src[indices[i]];
  }
}

int main() {
  Matrix3d X;
  X <<
    1, 2, 3,
    4, 5, 6,
    7, 8, 9;


  // RowVector3d X1 = X.rowwise()[0];  // No dice.

  auto X_rows = MakeRowView(X);
  cout << X_rows[0] << endl << endl;
  X_rows[2] *= 200;
  cout << X << endl << endl;

  const auto& X_const = X;
  auto X_rows_const = MakeRowView(X_const);
  cout << X_rows_const[1] << endl << endl;
  // X_rows_const[1] *= 10;

  MatrixXd X_sub(0, 3);
  auto X_sub_rows = MakeRowView(X_sub);
  X_sub_rows.resizeLike(X_rows, 2);
  select_indices({2, 0}, X_rows, X_sub_rows);
  cout << endl;
  cout << X_sub << endl << endl;

  auto X_cols = MakeColView(X);
  X_cols[1] /= 100.;
  cout << X << endl << endl;

  return 0;
}
