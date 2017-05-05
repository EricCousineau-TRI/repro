/*
Purpose: Same as the other matrix_stack* attempts, but this time, use tuples and explicit calling out of
hstack() and vstack() for concatenation.

*/

#include <utility>

namespace std {

template <int... Is>
struct index_sequence{ };

template <int I, int... Is>
struct make_index_sequence : public make_index_sequence<I-1, I-1, Is...> { };

template<int... Is>
struct make_index_sequence<0, Is...> : public index_sequence<Is...> { };

template<typename T>
using decay_t = typename decay<T>::type;

template<typename T>
using remove_cv_t = typename remove_cv<T>::type;

}  // namespace std




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


template<typename T>
using bare = std::remove_cv_t<std::decay_t<T>>;

template<typename... Args>
struct hstack_tuple;
template<typename... Args>
struct vstack_tuple;

template<typename T>
using is_hstack = is_specialization_of<T, hstack_tuple>;
template<typename T>
using is_vstack = is_specialization_of<T, vstack_tuple>;
template<typename T>
struct is_stack {
    static constexpr bool value = is_hstack<T>::value || is_vstack<T>::value;
};

template<typename Derived>
struct stack_detail {
    template<typename T>
    using is_scalar = std::is_convertible<T, typename Derived::Scalar>;

    static constexpr int
        TMatrix = 0,
        TScalar = 1,
        TStack = 2;

    template<typename T>
    struct type_index {
        static constexpr int value =
            is_scalar<T>::value ? TScalar : (
                is_stack<T>::value ? TStack :
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
            out.coeffRef(0, 0) = value;
        }
    };

    template<typename Stack>
    struct SubXpr<Stack, TStack> {
        Stack& value; // Mutable for now, for simplicity.
        SubXpr(Stack& value)
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
            value.assign(out);
        }
    };

    // More elegance???
    template<typename T>
    using SubXprAlias = SubXpr<bare<T>, type_index<bare<T>>::value>;

    template<typename T>
    static SubXprAlias<T> get_subxpr_helper(T&& x) {
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
    void check_size(XprType&& xpr, bool allow_resize) {
        if (xpr.rows() != m_rows || xpr.cols() != m_cols)
        {
            if (allow_resize)
                xpr.derived().resize(m_rows, m_cols);
            else
                // Can I include a message here?
                eigen_assert(xpr.rows() == m_rows && xpr.cols() == m_cols);
        }
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
    void assign(XprType&& xpr, bool allow_resize = false) {
        init_if_needed<Derived>();
        Base::check_size(xpr, allow_resize);

        int col = 0;
        auto f = [&](auto&& cur) {
            auto subxpr = stack_detail<Derived>::get_subxpr_helper(cur);
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
    void assign(XprType&& xpr, bool allow_resize = false) {
        init_if_needed<Derived>();
        Base::check_size(xpr, allow_resize);

        int row = 0;
        auto f = [&](auto&& cur) {
            auto subxpr = stack_detail<Derived>::get_subxpr_helper(cur);
            subxpr.assign(xpr.middleRows(row, subxpr.rows()));
            row += subxpr.rows();
        };
        Base::visit(f);
    }
};

// Actually leveraging std::forward_as_tuple
template<typename... Args>
hstack_tuple<Args&&...> hstack(Args&&... args) {
    return hstack_tuple<Args&&...>(std::forward<Args>(args)...);
}
template<typename... Args>
vstack_tuple<Args&&...> vstack(Args&&... args) {
    return vstack_tuple<Args&&...>(std::forward<Args>(args)...);
}

// Syntactic sugar
template<
    typename XprType,
    typename Stack,
    typename Derived = mutable_matrix_derived_type<XprType>,
    typename Cond = typename std::enable_if<is_stack<bare<Stack>>::value>::type
    >
void operator<<(XprType&& xpr, Stack&& stack) {
    // Permit resizing by default
    stack.assign(xpr, true);
}


using MatrixXs = Eigen::Matrix<string, Eigen::Dynamic, Eigen::Dynamic>;

void fill(MatrixXs& X, string prefix) {
    static const string hex = "0123456789abcdef";
    for (int i = 0; i < X.size(); ++i)
        X(i) = prefix + "[" + hex[i] + "]";
}

int main() {
    Eigen::Matrix<double, 1, 3> a;
    Eigen::Vector2d a1(1, 2);
    // Existing - better for non-ragged case.
    a << 10, a1.transpose();
    cout << "a: " << endl << a << endl << endl;
    hstack(10., a1.transpose()).assign(a);
    cout << "a: " << endl << a << endl << endl;

    Eigen::Matrix3d b;
    Eigen::Vector3d b1;
    b1.setConstant(1);
    Eigen::Matrix<double, 2, 3> b2;
    b2.setConstant(2);

    b << vstack(b1.transpose(), b2);
    cout << "b: " << endl << b << endl << endl;

    // Test for resize needed
    Eigen::VectorXd c;
    c << vstack(3, 2, 1);
    cout << "c: " << endl << c << endl << endl;

    // Test for resize for matrix
    Eigen::MatrixXd d;
    // hstack(a.transpose(), b, b1, c).assign(d); // Will assert at check_size
    d << hstack(a.transpose(), b, b1, c);
    cout << "d: " << endl << d << endl << endl;

    // Test nesting
    Eigen::Matrix2d e;
    e << hstack(vstack(1, 2), vstack(3, 4));
    cout << "e: " << endl << e << endl << endl;

    cout << endl << endl;

    // Use the test from `matrix_stack`, NumPy style
    MatrixXs
        A(1, 2), B(1, 2),
        C(2, 1),
        D(1, 3), E(1, 3),
        F(2, 4);
    fill(A, "A");
    fill(B, "B");
    fill(C, "C");
    fill(D, "D");
    fill(E, "E");
    fill(F, "F");

    string s1 = "s1";
    string s2 = "s2";
    string s3 = "s3";
    string s4 = "s4";

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

    MatrixXs X;
    X << vstack(
        hstack( vstack(A, B), C, vstack(D, E) ),
        hstack( F, vstack(hstack(s1, s2), hstack(s3, s4)) )
    );

    cout
        << "X: " << endl << X << endl << endl;

    return 0;
}
