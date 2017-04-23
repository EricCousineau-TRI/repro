// @ref https://github.com/hauptmech/eigen-initializer_list

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

using MatrixXc = Eigen::Matrix<char, Eigen::Dynamic, Eigen::Dynamic>;

int main() {
    MatrixXc A(1, 2), B(1, 2),
        C(2, 1),
        D(1, 3), E(1, 3),
        F(2, 6);
    A.setConstant('A');
    B.setConstant('B');
    C.setConstant('C');
    D.setConstant('D');
    E.setConstant('E');
    F.setConstant('F');

    cout
        << A << endl
        << B << endl
        << C << endl
        << D << endl
        << E << endl
        << F << endl;

    return 0;
}
