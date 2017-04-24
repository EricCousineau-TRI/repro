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
// namespace is_eigen_matrix_detail {
//     // These functions are never defined.
//     template <typename T>
//     std::true_type test(const Eigen::MatrixBase<T>*);

//     std::false_type test(...);
// }
// template <typename T>
// struct is_eigen_matrix
//     : public decltype(is_eigen_matrix_detail::test(std::declval<T*>()))
// { };
/* </snippet> */


/* <snippet from="./matrix_block.cc"> */
template<typename Derived>
Derived extract_mutable_derived_type(MatrixBase<Derived>&& value);
template<typename Derived>
Derived extract_mutable_derived_type(MatrixBase<Derived>& value);
template<typename T>
using mutable_matrix_derived_type = decltype(extract_mutable_derived_type(std::declval<T>()));
/* </snippet> */



// // Extend to compatible matrix types
// namespace is_convertible_eigen_matrix_detail {
//     template<typename Derived>
//     std::true_type test(const Eigen::MatrixBase<Derived>&);
//     std::false_type test(...);
// };
// template<typename T, typename DerivedTo>
// struct is_convertible_eigen_matrix
//     : public decltype(is_convertible_eigen_matrix_detail::test<DerivedTo>(std::declval<T>()))
// { };


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

template<typename... Args>
struct hstack_tuple;
template<typename... Args>
struct vstack_tuple;

template<typename Derived>
struct stack_detail {
    template<typename T>
    using is_scalar = std::is_convertible<T, typename Derived::Scalar>;
    template<typename T>
    using is_hstack = is_specialization_of<T, hstack_tuple>;
    template<typename T>
    using is_vstack = is_specialization_of<T, vstack_tuple>;

    static constexpr int
        TMatrix = 0,
        TScalar = 1,
        TStack = 2;

    template<typename T>
    struct type_index {
        static constexpr int value =
            is_scalar<T>::value ? TScalar : (
                is_hstack<T>::value || is_vstack<T>::value ? TStack :
                    TMatrix
                );
    };

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

    template<typename Stack>
    struct SubXpr<Stack, TStack> {
        const Stack& value;
        SubXpr(const Stack& value)
            : value(value) {
            value.template init_if_needed<Derived>();
        }
        int rows() {
            return value.m_rows;
        }
        int cols() {
            return value.m_cols;
        }
        template<typename AssignXprType>
        void assign(AssignXprType&& out) {
            cout << "assign stack" << endl;
            value.assign(out);
        }
    };

    template<typename T>
    using bare = std::remove_cv_t<std::decay_t<T>>;

    // More elegance???
    template<typename T>
    using SubXprAlias = SubXpr<bare<T>, type_index<bare<T>>::value>;

    template<typename T>
    static auto get_subxpr_helper(T&& x) {
        return SubXprAlias<T>(std::forward<T>(x));
    }
};

// First: Assume fixed-size, do assignment explicitly
// Next step: Collect into tuple, possibly accumulate size
// Then dispatch based on assignment
// operator<<

template<typename... Args>
struct stack_tuple {
    std::tuple<Args...> tuple;

    int m_rows {-1};
    int m_cols {-1};

    stack_tuple(Args&&... args)
        : tuple(std::forward<Args>(args)...)
    { }

    template<typename F>
    void visit(F&& f) {
        visit_tuple(std::forward<F>(f), tuple);
    }
    template<typename XprType>
    void resize_if_needed(XprType&& xpr) {
        if (xpr.rows() != m_rows || xpr.cols() != m_cols)
            xpr.derived().resize(m_rows, m_cols);
    }
};

// Define distinct types for identification
template<typename... Args>
struct hstack_tuple : public stack_tuple<Args...> {
    using Base = stack_tuple<Args...>;
    using Base::Base;
    using Base::m_cols;
    using Base::m_rows;

    template<typename Derived>
    void init_if_needed() {
        if (m_cols != -1) {
            eigen_assert(m_rows != -1);
            return;
        }
        // Need Derived type before use. Will defer until we 
        m_cols = 0;
        m_rows = -1;
        auto f = [&](auto&& cur) {
            auto subxpr = stack_detail<Derived>::get_subxpr_helper(cur);
            if (m_rows == -1)
                m_rows = subxpr.rows();
            else
                eigen_assert(subxpr.rows() == m_rows);
            m_cols += subxpr.cols();
        };
        Base::visit(f);
    }

    template<
        typename XprType,
        typename Derived = mutable_matrix_derived_type<XprType>
        >
    void assign(XprType&& xpr, bool resize = false) {
        init_if_needed<Derived>();
        if (resize)
            Base::resize_if_needed(xpr);

        int col = 0;
        auto f = [&](auto&& cur) {
            auto subxpr = stack_detail<Derived>::get_subxpr_helper(cur);
            cout << "col: " << col << endl;
            subxpr.assign(xpr.middleCols(col, subxpr.cols()));
            col += subxpr.cols();
        };
        Base::visit(f);
    }
};

template<typename... Args>
struct vstack_tuple : public stack_tuple<Args...> {
    using Base = stack_tuple<Args...>;
    using Base::Base;
    using Base::m_cols;
    using Base::m_rows;

    template<typename Derived>
    void init_if_needed() {
        if (m_cols != -1) {
            eigen_assert(m_rows != -1);
            return;
        }
        // Need Derived type before use. Will defer until we 
        m_cols = -1;
        m_rows = 0;
        auto f = [&](auto&& cur) {
            auto subxpr = stack_detail<Derived>::get_subxpr_helper(cur);
            if (m_cols == -1)
                m_cols = subxpr.cols();
            else
                eigen_assert(subxpr.cols() == m_cols);
            m_rows += subxpr.rows();
        };
        Base::visit(f);
    }

    template<
        typename XprType,
        typename Derived = mutable_matrix_derived_type<XprType>
        >
    void assign(XprType&& xpr, bool resize = false) {
        init_if_needed<Derived>();
        if (resize)
            Base::resize_if_needed(xpr);

        int row = 0;
        auto f = [&](auto&& cur) {
            auto subxpr = stack_detail<Derived>::get_subxpr_helper(cur);
            cout << "row: " << row << endl;
            subxpr.assign(xpr.middleRows(row, subxpr.rows()));
            row += subxpr.rows();
        };
        Base::visit(f);
    }
};

// Actually leveraging std::forward_as_tuple
template<typename... Args>
auto hstack(Args&&... args) {
    return hstack_tuple<Args&&...>(std::forward<Args>(args)...);
}
template<typename... Args>
auto vstack(Args&&... args) {
    return vstack_tuple<Args&&...>(std::forward<Args>(args)...);
}


int main() {
    Eigen::Matrix<double, 1, 3> a;
    Eigen::Vector2d a1(1, 2);
    hstack(10., a1.transpose()).assign(a);
    cout << a << endl;

    Eigen::Matrix3d b;
    Eigen::Vector3d b1;
    b1.setConstant(1);
    Eigen::Matrix<double, 2, 3> b2;
    b2.setConstant(2);

    vstack(b1.transpose(), b2).assign(b);
    cout << b << endl;

    Eigen::VectorXd c(3);
    vstack(3, 2, 1).assign(c);
    cout << c.transpose() << endl;

    return 0;
}
