/*
Purpose: sAme as the other matrix_stack* attempts, but this time, use tuples and explicit calling out of
hstack() and vstack() for concatenation.

*/
#include "cpp_quick/name_trait.h"

#include <string>
#include <iostream>
#include <memory>
#include <tuple>

#include <Eigen/Core>
#include <Eigen/Dense>

using std::cout;
using std::endl;
using std::string;
using std::unique_ptr;

using Eigen::MatrixBase;

/* <snippet from="http://stackoverflow.com/a/22726414/170413"> */
namespace is_eigen_matrix_detail {
    // These functions are never defined.
    template <typename T>
    std::true_type test(const Eigen::MatrixBase<T>*);

    std::false_type test(...);
}
template <typename T>
struct is_eigen_matrix
    : public decltype(is_eigen_matrix_detail::test(std::declval<T*>()))
{ };
/* </snippet> */


/* <snippet from="./matrix_block.cc"> */
template<typename Derived>
Derived extract_mutable_derived_type(MatrixBase<Derived>&& value);
template<typename Derived>
Derived extract_mutable_derived_type(MatrixBase<Derived>& value);
template<typename T>
using mutable_matrix_derived_type = decltype(extract_mutable_derived_type(std::declval<T>()));
/* </snippet> */



// Extend to compatible matrix types
namespace is_convertible_eigen_matrix_detail {
    template<typename Derived>
    std::true_type test(const Eigen::MatrixBase<Derived>&);
    std::false_type test(...);
};
template<typename T, typename DerivedTo>
struct is_convertible_eigen_matrix
    : public decltype(is_convertible_eigen_matrix_detail::test<DerivedTo>(std::declval<T>()))
{ };

// Specialize for non-matrix type
    // Issue: Need a mechanism to discrimnate based on a specific scalar type...
    // A failing case will be Matrix<Matrix<double, 2, 2>, ...> (e.g. tensor-ish stuff)

template<typename Derived>
struct stack_detail {
    template<typename T>
    using enable_if_scalar = std::enable_if<
        std::is_convertible<T, typename Derived::Scalar>::value,
        typename Derived::Scalar>;

    template<typename Scalar,
        typename Cond = typename enable_if_scalar<Scalar>::type>
    struct SubXpr {
        const Scalar& value;
        SubXpr(const Scalar& value)
            : value(value) { }
        int rows() {
            return 1;
        }
        int cols() {
            return 1;
        }
        template<typename AssignXprType>
        void assign(AssignXprType&& out) {
            out.coeffRef(0, 0) = value;
        }
    };

    template<typename XprType>
    struct SubXpr<XprType, void> {
        const XprType& value;
        SubXpr(const XprType& value)
            : value(value) { }
        int rows() {
            return value.rows();
        }
        int cols() {
            return value.rows();
        }
        template<typename AssignXprType>
        void assign(AssignXprType&& out) {
            out = value;
        }
    };
};

// First: Assume fixed-size
template<
    typename XprType,
    typename Derived = mutable_matrix_derived_type<XprType>,
    typename T1,
    typename... Args
    >
void hstack_into(XprType&& xpr, int col, T1&& t1, Args&&... args) {
    using detail = stack_detail<Derived>;
    using SubXpr = typename detail::template SubXpr<std::decay_t<T1>>;
    auto subxpr = SubXpr(t1);
    eigen_assert(xpr.rows() == subxpr.rows());
    int sub_cols = subxpr.cols();
    subxpr.assign(xpr.middleCols(col, sub_cols));
    hstack_into(xpr, col + sub_cols);
}

template<
    typename XprType,
    typename Derived = mutable_matrix_derived_type<XprType>
    >
void hstack_into(XprType&& xpr, int col) {
    eigen_assert(xpr.cols() == col);
}

int main() {
    // Eigen::Vector3d x;
    // hstack_into(x, 0, 1., 2., 3.);

    // cout << x.transpose() << endl;
    return 0;
}
