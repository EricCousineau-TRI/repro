#include <tuple>
#include <iostream>

#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

constexpr int TScalar = 0,
    TMatrix = 1,
    TTuple = 2;

/* <snippet from="https://bitbucket.org/martinhofernandes/wheels/src/default/include/wheels/meta/type_traits.h%2B%2B?fileviewer=file-view-default#cl-161"> */
// @ref http://stackoverflow.com/a/13101086/170413
//! Tests if T is a specialization of Template
template <typename T, template <typename...> class Template>
struct is_specialization_of : std::false_type {};
template <template <typename...> class Template, typename... Args>
struct is_specialization_of<Template<Args...>, Template> : std::true_type {};
/* </snippet> */

template <typename T>
struct is_eigen_matrix {
private:
    // See libstdc++, <type_traits>, __sfinae_types 
    typedef char good; // sizeof == 1
    struct bad { char value[2]; }; // sizeof == 2

    template <typename Derived>
    static good test(const MatrixBase<Derived>&);
    static bad test(...);
public:
    static constexpr bool value = sizeof(test(std::declval<T>())) == 1;
};

template <typename T>
struct is_tuple : public is_specialization_of<T, tuple> { };

template <typename T>
struct type_index_of {
    static constexpr int value = is_eigen_matrix<T>::value ? 
        TMatrix : (is_tuple<T>::value ?
            TTuple : TScalar);
};

int main() {
    cout << "scalar: " << type_index_of<double>::value << endl
         << "matrix: " << type_index_of<MatrixXd>::value << endl
         << "matrix: " << type_index_of<Vector3d>::value << endl
         << "tuple: " << type_index_of<tuple<int, int>>::value << endl;
    return 0;
}
