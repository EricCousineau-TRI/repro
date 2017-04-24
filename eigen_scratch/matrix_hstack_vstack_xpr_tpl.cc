/*
Purpose: sAme as the other matrix_stack* attempts, but this time, use tuples and explicit calling out of
hstack() and vstack() for concatenation.

*/
#include "cpp_quick/name_trait.h"
#include "cpp_quick/tuple_iter.h"

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


/* <snippet from="https://bitbucket.org/martinhofernandes/wheels/src/default/include/wheels/meta/type_traits.h%2B%2B?fileviewer=file-view-default#cl-161"> */
// @ref http://stackoverflow.com/a/13101086/170413
//! Tests if T is a specialization of Template
template <typename T, template <typename...> class Template>
struct is_specialization_of : std::false_type {};
template <template <typename...> class Template, typename... Args>
struct is_specialization_of<Template<Args...>, Template> : std::true_type {};
/* </snippet> */


// Specialize for non-matrix type
    // Issue: Need a mechanism to discrimnate based on a specific scalar type...
    // A failing case will be Matrix<Matrix<double, 2, 2>, ...> (e.g. tensor-ish stuff)

template<typename Derived>
struct stack_detail {
    template<typename T>
    using is_scalar = std::is_convertible<T, typename Derived::Scalar>;

    static constexpr int
        TMatrix = 0,
        TScalar = 1,
        THStack = 2,
        TVStack = 3;

    template<typename T>
    struct info {
    };

    // template<typename T>
    // using is_hstack = is_specialization_of<T, hstack_tuple> { };
    // template<typename T>
    // using is_vstack = is_specialization_of<T, vstack_tuple> { };

    // template<typename T>
    // using enable_if_scalar = std::enable_if<
    //     is_scalar<T>::value, typename Derived::Scalar>;


    template<typename XprType, int type = TMatrix>
    struct SubXpr {
        const XprType& value;
        SubXpr(const XprType& value)
            : value(value) { }
        int rows() {
            return value.rows();
        }
        int cols() {
            return value.cols();
        }
        template<typename AssignXprType>
        void assign(AssignXprType&& out) {
            cout << "assign Xpr: " << value << endl;
            out = value;
        }
    };

    template<typename Scalar>
    struct SubXpr<Scalar, TScalar> {
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
            cout << "assign scalar: " << value << endl;
            out.coeffRef(0, 0) = value;
        }
    };

    template<typename T>
    using bare = std::remove_cv_t<std::decay_t<T>>;

    // More elegance???
    template<typename T>
    using SubXprHelper = SubXpr<bare<T>, is_scalar<bare<T>>::value>;
};

// First: Assume fixed-size, do assignment explicitly
// Next step: Collect into tuple, possibly accumulate size
// Then dispatch based on assignment
// operator<<


template<typename... Args>
struct hstack_tuple : public std::tuple<Args...> {
    using Base = std::tuple<Args...>;
    using Base::Base;
};

// Actually leveraging std::forward_as_tuple
template<typename... Args>
auto make_hstack_tuple(Args&&... args) {
    return hstack_tuple<Args&&...>(std::forward<Args>(args)...);
}


template<
    typename XprType,
    typename Derived = mutable_matrix_derived_type<XprType>,
    typename... Args
    >
void hstack_into(XprType&& xpr, Args&&... args) {
    using detail = stack_detail<Derived>;
    int col = 0;
    auto f = [&](auto&& cur) {
        using T = std::remove_cv_t<std::decay_t<decltype(cur)>>;
        using SubXpr = typename detail::template SubXprHelper<T>;
        SubXpr subxpr(cur);
        eigen_assert(xpr.rows() == subxpr.rows());
        int sub_cols = subxpr.cols();
        cout << "col: " << col << endl;
        subxpr.assign(xpr.middleCols(col, sub_cols));
        col += sub_cols;
    };
    visit_args(f, std::forward<Args>(args)...);

    eigen_assert(col == xpr.cols());
}

template<
    typename XprType,
    typename Derived = mutable_matrix_derived_type<XprType>
    >
void vstack_into(XprType&& xpr, int row) {
    eigen_assert(xpr.rows() == row);
    cout << "done" << endl;
}

template<
    typename XprType,
    typename Derived = mutable_matrix_derived_type<XprType>,
    typename T1,
    typename... Args
    >
void vstack_into(XprType&& xpr, int row, T1&& t1, Args&&... args) {
    using detail = stack_detail<Derived>;
    using SubXpr = typename detail::template SubXprHelper<T1>;
    SubXpr subxpr(t1);
    eigen_assert(xpr.cols() == subxpr.cols());
    int sub_rows = subxpr.rows();
    cout << "row: " << row << endl;
    subxpr.assign(xpr.middleRows(row, sub_rows));
    vstack_into(xpr, row + sub_rows, std::forward<Args>(args)...);
}


int main() {
    Eigen::Vector2d a(1, 2);
    Eigen::Matrix<double, 1, 3> x;
    hstack_into(x,
        // 1., 2., 3.);
        10., a.transpose());

    auto t = make_hstack_tuple(10, a.transpose());

    cout << x << endl;

    Eigen::Matrix3d c;
    Eigen::Vector3d c1;
    c1.setConstant(1);
    Eigen::Matrix<double, 2, 3> c2;
    c2.setConstant(2);

    vstack_into(c, 0,
        c1.transpose(), c2);

    cout << c << endl;
    return 0;
}
