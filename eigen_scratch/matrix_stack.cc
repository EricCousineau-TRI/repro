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

    Challenge: Defer evaluation until the end
        Will figure that out later

*/

#include "cpp_quick/name_trait.h"

#include <string>
#include <iostream>

#include <Eigen/Dense>

using std::cout;
using std::endl;
using std::string;

class MatrixXc : public Eigen::Matrix<string, Eigen::Dynamic, Eigen::Dynamic> {
public:
    using Matrix::Matrix;

    using init_list = std::initializer_list<std::initializer_list<MatrixXc>>;

    MatrixXc(init_list list) {
        cout << "Stacked initialization" << endl;
    }
};

void fill(MatrixXc& X, string prefix) {
    static const string hex = "01234566789abcdef";
    for (int i = 0; i < X.size(); ++i)
        X(i) = prefix + hex[i];
}

int main() {
    MatrixXc A(1, 2), B(1, 2),
        C(2, 1),
        D(1, 3), E(1, 3),
        F(2, 6);
    fill(A, "A");
    fill(B, "B");
    fill(C, "C");
    fill(D, "D");
    fill(E, "E");
    fill(F, "F");

    cout
        << A << endl
        << B << endl
        << C << endl
        << D << endl
        << E << endl
        << F << endl;

    MatrixXc X = {
            { {{A}, {B}}, C, {{D}, {E}} },
            { F }
        };  

    return 0;
}
