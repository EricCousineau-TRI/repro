// Goal: See how easy it is to extend a Matrix's constructor
// Purpose: See if we can permit the semantics in `cpp_quick/composition_ctor.cc`
#include "cpp_quick/name_trait.h"

#include <string>
#include <iostream>

#include <Eigen/Dense>

using std::cout;
using std::endl;
using std::string;
using Eigen::Matrix;
using Eigen::MatrixXd;
using Eigen::VectorXd;

// Define toy inheritance

class MyMat : public MatrixXd {
public:
    // Specialize
    // NOTE: Must use greedy match to combat Eigen's template ctor
    template<typename T, typename Cond =
        typename std::enable_if<std::is_convertible<T, string>::value>::type>
    MyMat(const T& name)
        : MatrixXd() {
        string s(name);
        cout << "string: " << name << endl;
    }

    using MatrixXd::MatrixXd;

};

int main() {
    MyMat x("bob");

    return 0;
}
