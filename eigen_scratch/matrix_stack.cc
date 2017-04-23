// @ref https://github.com/hauptmech/eigen-initializer_list

// @ref ../cpp_quick/composition_ctor.cc
// @ref ./matrix_inheritance.cc

// Purpose: For fun

/*
Support matrix concatenation for a matrix such as:

  -------------
  | A |   | D |
  |---| C |---|
  | B |   | E |
  |-----------|
  |     F     |
  -------------

Achievable via:

    MATLAB:
    x =  [ [[A; B], C], [D; E];
                   F        ];

    Numpy:
    x = vstack(
            hstack( vstack(A, B), C, vstack(E, F) ),
            F
        )

    Possible? with initializer lists:
       {
         { {{A}, {B}}, C, {{D}, {E}} },
         { F} }
       }

    Grammar:

        initializer_list<Init> ==> hstack
        initializer_list<initializer_list<Init>> ==> vstack

    Achievable with composition construction (see ../composition_ctor.cc)

    Challenge: Defer evaluation until the end, do things efficiently, etc.
        Will figure that out later

*/

#include "cpp_quick/name_trait.h"

#include <string>
#include <iostream>

#include <Eigen/Core>
#include <Eigen/Dense>

using std::cout;
using std::endl;
using std::string;

template<typename Scalar>
class MatrixX : public Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> {
public:
    using Base = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    using Base::Base;

    // Initializing full matrix
    using col_initializer_list = std::initializer_list<MatrixX>; // list of columns
    using row_initializer_list = std::initializer_list<col_initializer_list>; // list of rows

    MatrixX(row_initializer_list row_list) {
        int rows = 0;
        int cols = -1;

        // Get variable for clarity
        auto& X = Base::derived(); // Need 'this->' or other spec with class being templated

        // First review size
        for (const auto& col_list : row_list) {
            // Construct row:
            // @require: All items have same number of rows
            // @require: Columns be equal to overall # of columns
            int row_rows = -1;
            int row_cols = 0;
            for (const auto& item : col_list) {
                int item_rows = item.rows();
                int item_cols = item.cols();
                if (row_rows == -1)
                    row_rows = item_rows;
                else
                    // Already set, must match
                    eigen_assert(item_rows == row_rows);
                row_cols += item_cols;
            }
            if (cols == -1)
                cols = row_cols;
            else
                // Already set, must match
                eigen_assert(row_cols == cols);
            rows += row_rows;
        }

        if (cols == -1)
            // Ensure that we have a valid size
            cols = 0;

        // We now have our desired size
        X.resize(rows, cols);

        // Now fill in the data
        // We know that our data is good, no further checks needed
        int r = 0;
        for (const auto& col_list : row_list) {
            int c = 0;
            int row_rows = 0;
            for (const auto& item : col_list) {
                int item_rows = item.rows();
                int item_cols = item.cols();
                X.block(r, c, item_rows, item_cols) = item;
                row_rows = item_rows;
                c += item_cols;
            }
            r += row_rows;
        }
    }

    // // Initializing a row
    // // Challenge: Permitting initializer lists for column Vectors, while avoiding
    // // ambiguity...
    // // Solution? Intepret scalar initializer list as column initializing if its column
    // // dimension is static and singleton.
    // // This should permit scalar lists within larger lists to still be interpreted as
    // // row initialization

    // using scalar_initializer_list = std::initializer_list<Scalar>; // row of scalars

    // // Issue: Makes constructor ambiguous...
    // MatrixX(scalar_initializer_list row) {
    //     auto& X = Base::derived();
    //     X.resize(row.size());
    //     for (int i = 0; i < row.size(); ++i)
    //         X(i) = row[i];
    // }

    // Permit scalars
    MatrixX(const Scalar& s) {
        // TODO: Somehow enable static-sized 1x1 matrices? Expression template magic?
        auto& X = Base::derived();
        X.resize(1, 1);
        X(0) = s;
    }
};


using scalar_type = string;

using MatrixXc = MatrixX<scalar_type>;

void fill(MatrixXc& X, scalar_type prefix) {
    static const scalar_type hex = "0123456789abcdef";
    for (int i = 0; i < X.size(); ++i)
        X(i) = prefix + "[" + hex[i] + "]";
}

int main() {
    MatrixXc A(1, 2), B(1, 2),
        C(2, 1),
        D(1, 3), E(1, 3),
        F(2, 4);
    fill(A, "A");
    fill(B, "B");
    fill(C, "C");
    fill(D, "D");
    fill(E, "E");
    fill(F, "F");

    scalar_type s1 = "s1";
    scalar_type s2 = "s2";
    scalar_type s3 = "s3";
    scalar_type s4 = "s4";

    cout
        << "A: " << endl << A << endl << endl
        << "B: " << endl << B << endl << endl
        << "C: " << endl << C << endl << endl
        << "D: " << endl << D << endl << endl
        << "E: " << endl << E << endl << endl
        << "F: " << endl << F << endl << endl;

    cout
        << "Scalars: "
        << s1 << ", " << s2 << ", " << s3 << ", " << s4
        << endl << endl;

    MatrixXc X = {
            { {{A}, {B}}, C, {{D}, {E}} },
            { F, {{s1, s2}, {s3, s4}} }
        };

    cout
        << "X: " << endl << X << endl << endl;

    return 0;
}
