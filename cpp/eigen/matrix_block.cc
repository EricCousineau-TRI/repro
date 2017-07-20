// Purpose: Figure out how to generically use different expressions, to be compatible with assigning to blocks and what not

// Better use of decltype?
// @ref http://stackoverflow.com/a/22726414/170413

// TODO: Determine if this is possible just with inference?
//   is_base_of<MatrixBase<Derived>, Derived>
// Would need additional decay_t, and ensure that it is non-const
// See: http://en.cppreference.com/w/cpp/types/is_reference

#include "cpp/name_trait.h"

#include <string>
#include <iostream>

#include <Eigen/Core>
#include <Eigen/Dense>

using std::cout;
using std::endl;
using std::string;

using Eigen::DenseBase;
using Eigen::Block;
using Eigen::MatrixXd;
using Eigen::Matrix3d;
using Eigen::MatrixBase;
using Eigen::Vector2d;
using Eigen::Vector3d;
using Eigen::VectorXd;

/**
 * For extracting Derived type with const-lvalue for maximum type affinity
 * Leverage compiler's polymorphism semantics so we don't have to deal with it
 * 
 * Defined such that only lvalue and rvalue references are compatible.
 * const-lvalue references are strictly prohibited by construction.
 * @ref http://stackoverflow.com/questions/32282705/a-failure-to-instantiate-function-templates-due-to-universal-forward-reference
 *
 * @note Cannot be used except for in the context of decltype()!
 */
template<typename Derived>
Derived extract_mutable_derived_type(MatrixBase<Derived>&& value);
template<typename Derived>
Derived extract_mutable_derived_type(MatrixBase<Derived>& value);

/**
 * Return the derived type of a given matrix, leveraging extract_derived
 * to bypass intermediate polymorphism, but ensure that the expression is non-const
 * Leverage matrix_derived_type with decltype() to implement SFINAE, such that
 * you can (a) use a fully resolved type, which then means you can (b) use
 * perfect forwarding to obtain lvalue or rvalue references (e.g., with Blocks),
 * while excluding const-lvalue types.
 *
 * TODO: Figure out how to idenfity a type which fails SFINAE. Use AssertionChecker pattern?
 */
template<typename T>
using mutable_matrix_derived_type = decltype(extract_mutable_derived_type(std::declval<T>()));

/*
// Can't get this to work as intended...

// Only use this if you will only be writing to matrix types, and wish to see the failures
// Successful case
template<typename T,
    typename Derived = mutable_matrix_derived_type<T>>
struct mutable_matrix_derived_type_with_check_impl {
    using type = Derived;
};
// Failure case
template<typename Derived>
struct mutable_matrix_derived_type_with_check_impl<Derived, void> {
    using type = void;
    static_assert(!std::is_same<Derived, Derived>::value, "Type is either non-matrix or an immutable matrix");
};
*/

/**
 * Unused, but for demonstration. Can also used for immutable matrices
 *
 * @note Will permit lvalue, const-lvalue, and rvalue references, and will permit all of
 * these types during template substitution.
 */
template<typename Derived>
Derived extract_derived(const MatrixBase<Derived>& value) {
    return std::declval<Derived>();
}
template<typename T>
using matrix_type = decltype(exract_derived(std::declval<T>()));
/* End Demonstration */


// Using mutable_matrix_derived_type:
template<typename XprType,
    typename Derived = mutable_matrix_derived_type<XprType>>
auto&& fill(XprType&& x) {
    // cannot use decltype(x) on return type?
    x.setConstant(1);
    return std::forward<XprType>(x);
}

template<typename DerivedA, typename XprTypeB,
    typename DerivedB = mutable_matrix_derived_type<XprTypeB>>
        // mutable_matrix_derived_type<XprTypeB>> // Use for SFINAE
void evalTo(const MatrixBase<DerivedA>& x, XprTypeB&& y) {
    // Do a lot of complex operations
    // Leverage direct typename to use perfect forwarding
    // (but constrained to mutable references)
    y += x;
}

/*
// -- HACK ---
// AVOID IF YOU CAN
// Obtain lvalue reference from rvalue reference.
// Only use if you know what you are doing!

template<typename T>
T& to_lvalue(T&& x) {
    return static_cast<T&>(x);
}
template<typename T>
T&& to_rvalue(T& x) {
    return static_cast<T&&>(x);
}

template<typename Derived>
void fillHackC98(MatrixBase<Derived> const& x_hack) {
    // C98: Be wary!!! Using hack from:
    // https://eigen.tuxfamily.org/dox/TopicFunctionTakingEigenTypes.html#title3
    auto& x = const_cast<MatrixBase<Derived>&>(x_hack);
    x.setConstant(1);
}

template<typename Derived>
MatrixBase<Derived>& fillHack(MatrixBase<Derived>& x) {
    cout << "lvalue" << endl;
    x.setConstant(1);
    return x;
}
template<typename Derived>
MatrixBase<Derived>&& fillHack(MatrixBase<Derived>&& x) {
    cout << "rvalue" << endl;
    // Secondary hack (cleaner due to not fiddling with const, but still a hack)
    // Cleaner alternative: Reimplement the functionality
    fill(to_lvalue(x));
    // If using this hack, and you are returning a reference, you should return to rvalue
    // NOTE: Perhaps try avoiding `std::move`, as that may confuse people reviewing the code.
    // Rather, explicitly show that you are returning to an rvalue (if need be)
    return to_rvalue(x);
}
*/

// Implicit rvalue
Matrix3d example() {
    Matrix3d x;
    x.setConstant(10);
    return x;
}

int main() {
    MatrixXd A(2, 2);
    fill(A); // lvalue, dynamic
    cout << "A: " << endl << A << endl << endl;

    Matrix3d B;
    fill(B); // lvalue, static
    cout << "B: " << endl << B << endl << endl;

    MatrixXd C(3, 2);
    C.setZero();
    fill(C.block(0, 0, 2, 2)) // rvalue
        .coeffRef(0, 0) = 20; // Chain a statement afterwards, valid for rvalue
    cout << "C: " << endl << C << endl << endl;

    cout << "Explicit rvalue example: " << endl
         << fill(example()) << endl;

    // Example useful stuff
    VectorXd y(5);
    y.setConstant(1);
    evalTo(VectorXd::Ones(5), y);
    evalTo(VectorXd::Ones(3), y.head(3));
    evalTo(5 * Vector2d::Ones(), y.tail(2));

    // // Get prettier error?
    // evalTo(5 * Vector2d::Ones(), 5);

    cout << "y: " << y.transpose() << endl;

    return 0;
}
