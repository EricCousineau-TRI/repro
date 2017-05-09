/*
Purpose: Same as the other matrix_stack* attempts, but this time, use tuples and explicit calling out of
hstack() and vstack() for concatenation.

*/

#include <utility>

#if __cplusplus <= 201103L
namespace std {

template <size_t... Is>
struct index_sequence{ };

template <size_t I, size_t... Is>
struct make_index_sequence : public make_index_sequence<I-1, I-1, Is...> { };

template<size_t... Is>
struct make_index_sequence<0, Is...> : public index_sequence<Is...> { };

template<typename T>
using decay_t = typename decay<T>::type;

template<typename T>
using remove_cv_t = typename remove_cv<T>::type;

}  // namespace std
#endif


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

template <typename Scalar, int Rows, int Cols>
struct name_trait<Eigen::Matrix<Scalar, Rows, Cols>> {
    static std::string dim_name(int dim) {
        if (dim == Eigen::Dynamic)
            return "X";
        else
            return std::to_string(dim);
    }
    static std::string name() {
        return "Matrix<" + name_trait<Scalar>::name() + ", " + dim_name(Rows) + ", " + dim_name(Cols) + ">";
    }
};

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


template<typename Derived>
Derived extract_derived_type(const MatrixBase<Derived>& value);
template<typename T>
using matrix_derived_type = decltype(extract_derived_type(std::declval<T>()));


template <typename T>
struct is_eigen_matrix {
private:
    // See libstdc++, <type_traits>, __sfinae_types 
    // Modified to use true_type, false_type
    template <typename Derived>
    static std::true_type test(const MatrixBase<Derived>&);
    static std::false_type test(...);
public:
    static constexpr bool value = decltype(test(std::declval<T>()))::value;
};

// Quick binary operator reduction
// @note This permits a single argument, assuming idempotent stuff is OK

// Base case for specialization
template <template <int,int> class Op, int... Cs>
struct binary_reduction;
// Single-argument (unary) case
template <template <int,int> class Op, int A>
struct binary_reduction<Op, A> {
    static constexpr int value = A;
};
// 2+ argument case
template <template <int,int> class Op, int A, int B, int... Cs>
struct binary_reduction<Op, A, B, Cs...> {
    static constexpr int value = binary_reduction<Op, Op<A, B>::value, Cs...>::value;
};

template <int A, int B>
struct eigen_dim_op_sum {
    static constexpr int value = (A == Eigen::Dynamic || B == Eigen::Dynamic) ? Eigen::Dynamic : A + B;
};

template <int A, int B>
struct eigen_dim_op_eq {
    static_assert((A == Eigen::Dynamic || B == Eigen::Dynamic) || A == B,
        "To use compile-time concatenation, dimensions must match or be dynamic.");
    static constexpr int value = (A == Eigen::Dynamic || B == Eigen::Dynamic) ? Eigen::Dynamic : A;
};


template <int... Cs>
using eigen_dim_sum = binary_reduction<eigen_dim_op_sum, Cs...>;
template <int... Cs>
using eigen_dim_eq = binary_reduction<eigen_dim_op_eq, Cs...>;



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
using bare_t = std::remove_cv_t<std::decay_t<T>>;

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



// Quick attempt to infer types
template <typename T>
struct infer_scalar {
private:
    static constexpr int TOther = 0, TMatrix = 1, TStack = 2;
    static constexpr int type_index =
        is_eigen_matrix<T>::value ? TMatrix : (
            is_stack<T>::value ? TStack : TOther
        );
    // Need to introduce type to permit defining specialization w/in struct
    template <typename Scalar, int TIndex = TOther>
    struct get { using type = Scalar; };
    template <typename XprType>
    struct get<XprType, TMatrix> { using type = typename matrix_derived_type<XprType>::Scalar; };
    template <typename Stack>
    struct get<Stack, TStack> { using type = typename Stack::ScalarInferred; };
public:
    using type = typename get<T, type_index>::type;
};

template <typename T>
using infer_scalar_bare_t = typename infer_scalar<bare_t<T>>::type;



template<typename Scalar>
struct stack_detail {
    template<typename T>
    using is_scalar = std::is_convertible<T, Scalar>;

    static constexpr int
        TMatrix = 0,
        TScalar = 1,
        TStack = 2;

    // Specialize this to the context of a stack, where scalars may be other matrices
    template<typename T>
    struct type_index {
        // Check for scalars first, to avoid prematurely assuming MatrixXd, in the context of MatrixX<MatriXd>, is a scalar.
        static constexpr int value =
            is_scalar<T>::value ? TScalar : (
                is_stack<T>::value ? TStack :
                    TMatrix
                );
    };

    template<typename XprType, int type = TMatrix>
    struct SubXpr {
        const XprType& value;
        // This is used in tye default case. We should throw an error if this is not an actual matrix.
        static_assert(is_eigen_matrix<XprType>::value, "This type is not a Eigen Matrix, compatible scalar, nor a stack type.");

        struct dim_traits {
            static constexpr int ColsAtCompileTime = XprType::ColsAtCompileTime;
            static constexpr int RowsAtCompileTime = XprType::RowsAtCompileTime;
        };

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

    template<typename SubScalar>
    struct SubXpr<SubScalar, TScalar> {
        const SubScalar& value;

        struct dim_traits {
            static constexpr int ColsAtCompileTime = 1;
            static constexpr int RowsAtCompileTime = 1;
        };

        SubXpr(const SubScalar& value)
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

        using dim_traits = typename Stack::template dim_traits<Scalar>;

        SubXpr(Stack& value)
            : value(value) {
            value.template init_if_needed<Scalar>();
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
    using SubXprAlias = SubXpr<bare_t<T>, type_index<bare_t<T>>::value>;

    template<typename T>
    static SubXprAlias<T> get_subxpr_helper(T&& x) {
        return SubXprAlias<T>(std::forward<T>(x));
    }
};


template<typename... Args>
struct stack_tuple {
    using TupleType = std::tuple<Args...>;
    TupleType tuple;

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

    // For deducing final type
    template <typename Scalar,
        template <int...> class RowDimOp,
        template <int...> class ColDimOp>
    struct dim_traits {
        template <typename T>
        using SubXprAlias = typename stack_detail<Scalar>::template SubXprAlias<T>;
        template <typename T>
        using SubDimTraits = typename SubXprAlias<T>::dim_traits;

        static constexpr int ColsAtCompileTime = ColDimOp<SubDimTraits<Args>::ColsAtCompileTime...>::value;
        static constexpr int RowsAtCompileTime = RowDimOp<SubDimTraits<Args>::RowsAtCompileTime...>::value;

        using FinishedType = typename Eigen::Matrix<Scalar, RowsAtCompileTime, ColsAtCompileTime>;
    };

    // TODO: Is there a better way to implement this?
    using first_type = decltype(std::get<0>(std::declval<TupleType>()));

    using ScalarInferred = infer_scalar_bare_t<first_type>;
};


// Define distinct types for identification
template<typename... Args>
struct hstack_tuple : public stack_tuple<Args...> {
    using Base = stack_tuple<Args...>;
    using Base::Base;
    using Base::m_cols;
    using Base::m_rows;

    template<typename Scalar>
    void init_if_needed() {
        if (m_cols != -1) {
            eigen_assert(m_rows != -1);
            return;
        }
        // Need Derived type before use. Will defer until we 
        m_cols = 0;
        m_rows = -1;
        InitFunctor<Scalar> f {m_rows, m_cols};
        Base::visit(f);
    }

    template <typename Scalar>
    struct InitFunctor {
        // Context
        int& m_rows;
        int& m_cols;
        // Method
        template <typename T>
        void operator()(T&& cur) {
            auto subxpr = stack_detail<Scalar>::get_subxpr_helper(cur);
            if (m_rows == -1)
                m_rows = subxpr.rows();
            else
                eigen_assert(subxpr.rows() == m_rows);
            m_cols += subxpr.cols();
        }
    };

    template<
        typename XprType,
        typename Derived = mutable_matrix_derived_type<XprType>
        >
    XprType&& assign(XprType&& xpr, bool allow_resize = false) {
        using Scalar = typename Derived::Scalar;
        init_if_needed<Scalar>();
        Base::check_size(xpr, allow_resize);

        int col = 0;
        AssignFunctor<XprType, Scalar> f {std::forward<XprType>(xpr), col};
        Base::visit(f);
        return std::forward<XprType>(xpr);
    }

    template <typename XprType, typename Scalar>
    struct AssignFunctor {
        // Context
        XprType&& xpr;
        int& col;
        // Method
        template <typename T>
        void operator()(T&& cur) {
            auto subxpr = stack_detail<Scalar>::get_subxpr_helper(cur);
            subxpr.assign(xpr.middleCols(col, subxpr.cols()));
            col += subxpr.cols();
        }
    };

    template <typename Scalar>
    using dim_traits = typename Base::template dim_traits<Scalar, eigen_dim_eq, eigen_dim_sum>;

    template <typename Scalar = typename Base::ScalarInferred,
        typename FinishedType = typename dim_traits<Scalar>::FinishedType>
    FinishedType finished() {
        return assign(FinishedType(), true);
    }
};

template<typename... Args>
struct vstack_tuple : public stack_tuple<Args...> {
    using Base = stack_tuple<Args...>;
    using Base::Base;
    using Base::m_cols;
    using Base::m_rows;

    template<typename Scalar>
    void init_if_needed() {
        if (m_cols != -1) {
            eigen_assert(m_rows != -1);
            return;
        }
        // Need Derived type before use. Will defer until we 
        m_cols = -1;
        m_rows = 0;
        InitFunctor<Scalar> f {m_rows, m_cols};
        Base::visit(f);
    }

    template <typename Scalar>
    struct InitFunctor {
        // Context
        int& m_rows;
        int& m_cols;
        // Method
        template <typename T>
        void operator()(T&& cur) {
            auto subxpr = stack_detail<Scalar>::get_subxpr_helper(cur);
            if (m_cols == -1)
                m_cols = subxpr.cols();
            else
                eigen_assert(subxpr.cols() == m_cols);
            m_rows += subxpr.rows();
        }
    };

    template<
        typename XprType,
        typename Derived = mutable_matrix_derived_type<XprType>
        >
    XprType&& assign(XprType&& xpr, bool allow_resize = false) {
        using Scalar = typename Derived::Scalar;
        init_if_needed<Scalar>();
        Base::check_size(xpr, allow_resize);

        int row = 0;
        AssignFunctor<XprType, Scalar> f {std::forward<XprType>(xpr), row};
        Base::visit(f);
        return std::forward<XprType>(xpr);
    }

    template <typename XprType, typename Scalar>
    struct AssignFunctor {
        // Context
        XprType&& xpr;
        int& row;
        // Method
        template <typename T>
        void operator()(T&& cur) {
            auto subxpr = stack_detail<Scalar>::get_subxpr_helper(cur);
            subxpr.assign(xpr.middleRows(row, subxpr.rows()));
            row += subxpr.rows();
        }
    };


    template <typename Scalar>
    using dim_traits = typename Base::template dim_traits<Scalar, eigen_dim_sum, eigen_dim_eq>;

    template <typename Scalar = typename Base::ScalarInferred,
        typename FinishedType = typename dim_traits<Scalar>::FinishedType>
    FinishedType finished() {
        return assign(FinishedType(), true);
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
    typename Cond = typename std::enable_if<is_stack<bare_t<Stack>>::value>::type
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

template <typename T>
std::string type_name_of(const T&) {
    return name_trait<T>::name();
}

int main() {
    Eigen::Matrix<double, 1, 3> a;
    Eigen::Vector2d a1(1, 2);
    // Existing - better for non-ragged case.
    a << 10, a1.transpose();
    cout << "a: " << type_name_of(a) << endl << a << endl << endl;
    hstack(10., a1.transpose()).assign(a);
    cout << "a: " << endl << a << endl << endl;

    auto a_tmp = hstack(10., a1.transpose()).finished();
    cout << "a_tmp: " << type_name_of(a_tmp) << endl << a_tmp << endl << endl;

    // Check dynamic sizing
    auto ax_tmp = hstack(10., Eigen::VectorXd(a1).transpose()).finished();
    cout << "ax_tmp: " << type_name_of(ax_tmp) << endl << ax_tmp << endl << endl;

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
    // Need to explicitly specify types here, or use .finished<double>()
    auto e_tmp = hstack(vstack(1., 2.), vstack(3., 4.)).finished();
    cout << "e_tmp: " << type_name_of(e_tmp) << endl << e_tmp << endl << endl;

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

    auto X_tmp = vstack(
            hstack( vstack(A, B), C, vstack(D, E) ),
            hstack( F, vstack(hstack(s1, s2), hstack(s3, s4)) )
        ).finished();
    cout << "X_tmp: " << type_name_of(X_tmp) << endl << X_tmp << endl << endl;

    return 0;
}
