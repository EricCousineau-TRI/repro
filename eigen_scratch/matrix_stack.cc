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
#include <memory>

#include <Eigen/Core>
#include <Eigen/Dense>

using std::cout;
using std::endl;
using std::string;
using std::unique_ptr;

/*
TODO: Attempt using analog to Eigen/src/Core/CommaInitializer.h
Goal:
Permit something like
  X << {{A}, {B}};   // To maintain Xpr template-ness
In copy ctor, could then do
  derived() << {{A}, {B}}
Challenge:
  Need to enable the abilitity to compile-time store expressions...
  Possibly concatenating tuples... Which will be recursive...
  Or... Just return expressions?
*/

/*
NOTE:
To avoid the need for temporaries, could consider using templates with internal
linkage, at least once GCC / clang actually suppport it:
@ref https://gcc.gnu.org/bugzilla/show_bug.cgi?id=52036

Then we could do something like;
template<typename XprType, DenseBase<Derived>& block, typename Derived>
class XprNode { ... };

and then invoke magically...
    Top-level
        XprNode<XprType, *this>(row_list) 
    Sub-expression
        ImplSubXprNode<XprType, *this>(row_list)
*/

template<typename XprType>
class XprNode {
public:
    using Scalar = XprType::Scalar;
    // // Explicitly constrain to a block expression... ???
    // using Block = XprType::Block;

    // Initializing full matrix
    using col_initializer_list = std::initializer_list<XprType>; // list of columns
    using row_initializer_list = std::initializer_list<col_initializer_list>; // list of rows

    // Constructors
    XprNode(const Scalar& s)
        : impl(new ImplScalarCRef(s))
    { }
    XprNode(Scalar&& s)
        : impl(new ImplScalarRRef(s))
    { }
    template<typename Derived>
    XprNode(const DenseBase<Derived>& o)
        : impl(new ImplDenseCRef<Derived>(o))
    { }
    template<typename Derived>
    XprNode(DenseBase<Derived>&& o)
        : impl(new ImplDenseRRef<Derived>(o))
    { }
    XprNode(row_initializer_list row_list)
        : impl(new ImplSubXprNode(row_list))
    { }

    int rows() const {
        return impl->rows();
    }
    int cols() const {
        return impl->cols();
    }

    template<typename Derived>
    void apply(int r, int c, DenseBase<Derived>& block) {
        eigen_assert(block.rows() == rows() && block.cols() == cols());
        // :(
        // Can't figure out how to get rid of polymorphism, or
        // how to even virtualize this templated method...
        // NOTE: Can get ride of polymorphism...
        // But would need top-level determination of rows / cols
        if (auto p = try_cast<ImplScalarCRef>()) {
            apply_scalar(p, block);
        } else if (auto p = try_cast<ImplScalarRRef>()) {
            apply_scalar(p, block);
        } else if (auto p = try_cast<ImplDenseCRef>()) {
            apply_dense(p, block);
        } else if (auto p = try_cast<ImplDenseRRef>()) {
            apply_dense(p, block);
        } else if (auto p = try_cast<ImplSubXprNode>()) {
            apply_subxpr(p, block);
        }
    }
private:
    typename<typename ImplType>
    ImplType* try_cast() {
        return dynamic_cast<ImplType*>(impl.get());
    }

    template<typename ImplType, typename Derived>
    void apply_scalar(ImplType* p, DenseBase<Derived>& block) {
        block.coeffRef(0, 0) = p->value;
    }

    template<typename ImplType, typename Derived>
    void apply_dense(ImplType* p, DenseBase<Derived>& block) {
        block = p->value;
    }

    struct Impl {
        // How to avoid 'virtual'???
        // Initializer list with differred construction???
        // Tuple???
        virtual int rows() const = 0;
        virtual int cols() const = 0;
        // // Not sure how to do this in a bottom-up fashion...
        // virtual void apply(int r, int c, XprType::BlockExpr&) const = 0;
    };

    // Try to reference the scalar (if it is an expensive type)
    struct ImplScalarCRef : public Impl {
        const Scalar& value;
        ImplScalarCRef(const Scalar& value)
            : value(value)
        { }
        int rows() const { return 1; }
        int cols() const { return 1; }
    };
    // Temporary expression, need to store
    struct ImplScalarRRef : public Impl {
        Scalar value;
        ImplScalarRRef(Scalar&& value)
            : value(std::move(value))
        { }
        int rows() const { return 1; }
        int cols() const { return 1; }
    };

    // Try to reference the dense expression
    template<typename Derived>
    struct ImplDenseCRef : public Impl {
        const DenseBase<Derived>& value;
        ImplDenseCRef(const DenseBase<Derived>& value)
            : value(value)
        { }
        int rows() const { return value.rows(); }
        int cols() const { return value.cols(); }
    };
    // Store temporary
    // HOPE: That this temporary is an expression template type,
    // and its evaluation is defferred until 'apply'
    template<typename Derived>
    struct ImplDenseRRef : public Impl {
        DenseBase<Derived> value;
        ImplDenseCRef(DenseBase<Derived>&& value)
            : s(std::move(value))
        { }
        int rows() const { return value.rows(); }
        int cols() const { return value.cols(); }
    };
    // Store SubXpr
    struct ImplSubXprNode : public Impl {
        XprNode value;
        int m_rows;
        int m_cols;
        ImplSubXprNode(row_intializer_list row_list)
            : value(row_list), m_rows(0), m_cols(-1) {
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
                if (m_cols == -1)
                    m_cols = row_cols;
                else
                    // Already set, must match
                    eigen_assert(row_cols == m_cols);
                rows += row_rows;
            }
            if (m_cols == -1) {
                // Ensure that we have a valid size
                m_cols = 0;
            }
        }
        int rows() const { return m_rows; }
        int cols() const { return m_cols; }
    };
    
    unique_ptr<XprImpl> impl; // :(
};

template<typename Scalar>
class MatrixX : public Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> {
public:
    using Base = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    using Base::Base;

    // MatrixX(row_initializer_list row_list) {

    //     // Get variable for clarity
    //     auto& X = Base::derived(); // Need 'this->' or other spec with class being templated



    //     // We now have our desired size
    //     X.resize(rows, cols);

    //     // Now fill in the data
    //     // We know that our data is good, no further checks needed
    //     int r = 0;
    //     for (const auto& col_list : row_list) {
    //         int c = 0;
    //         int row_rows = 0;
    //         for (const auto& item : col_list) {
    //             int item_rows = item.rows();
    //             int item_cols = item.cols();
    //             X.block(r, c, item_rows, item_cols) = item;
    //             row_rows = item_rows;
    //             c += item_cols;
    //         }
    //         r += row_rows;
    //     }
    // }

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

    // // Permit scalars
    // MatrixX(const Scalar& s) {
    //     // TODO: Somehow enable static-sized 1x1 matrices? Expression template magic?
    //     auto& X = Base::derived();
    //     X.resize(1, 1);
    //     X(0) = s;
    // }
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

    // MatrixXc X = {
    //         { {{A}, {B}}, C, {{D}, {E}} },
    //         { F, {{s1, s2}, {s3, s4}} }
    //     };

    // cout
    //     << "X: " << endl << X << endl << endl;

    return 0;
}
