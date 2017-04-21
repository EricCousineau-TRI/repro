#include <iostream>
#include <string>
using std::string;
using std::cout;
using std::endl;

#include "name_trait.h"

template<typename Cond>
struct AssertionChecker : Cond {
    static_assert(Cond::value, "Condition failed");
};

template<typename T>
struct is_good : std::true_type { };

// Make it fail only on double[2]
template<>
struct is_good<double[2]> : std::false_type { };


/* Unfriendly version: just return false
There will be fewer errors, but they won't pinpoint which type made the
std::enable_if<> fail
*/

/* --- Unfriendly Error ---
cpp_quick/tpl_check_errors.cc:110:5: error: no matching function for call to 'my_func'
    my_func(int{},
    ^~~~~~~
cpp_quick/tpl_check_errors.cc:99:29: note: candidate template ignored: disabled by 'enable_if' [with Args = <int, std::__cxx11::basic_string<char>, name_trait_list<int, std::__cxx11::basic_string<char>, double>, double (&)[2], double>]
    typename std::enable_if<is_all_good<Args...>::value>::type>
                            ^
*/




/* Friendly(ish) version: Make assertion fail as closely as possible
This explicit denotes that it failed on 

NOTE: This only useful if you want to force an error when substitution
failure occurring means that there will be an error.

If you are trying to debug why you are not encountering the overload you
want... Well... not sure how to handle that.

@ref http://stackoverflow.com/a/13366183/7829525
*/

/* --- Friendly Error ---
cpp_quick/tpl_check_errors.cc:11:5: error: static_assert failed "Condition failed"
    static_assert(Cond::value, "Condition failed");
    ^             ~~~~~~~~~~~
cpp_quick/tpl_check_errors.cc:52:25: note: in instantiation of template class 'AssertionChecker<is_good<double [2]> >' requested here
struct is_good_custom : AssertionChecker<is_good<T>> { };
                        ^
cpp_quick/tpl_check_errors.cc:96:7: note: in instantiation of template class 'is_good_custom<double [2]>' requested here
    : is_good_custom<typename std::remove_reference<T>::type> { };
      ^
cpp_quick/tpl_check_errors.cc:88:9: note: in instantiation of template class 'is_all_good<double (&)[2]>' requested here
        is_all_good<T>::value ?
        ^
cpp_quick/tpl_check_errors.cc:89:13: note: in instantiation of template class 'is_all_good<double (&)[2], double>' requested here
            is_all_good<Args...>::value : false;
            ^
cpp_quick/tpl_check_errors.cc:89:13: note: in instantiation of template class 'is_all_good<name_trait_list<int, std::__cxx11::basic_string<char>, double>, double (&)[2], double>' requested here
cpp_quick/tpl_check_errors.cc:89:13: note: in instantiation of template class 'is_all_good<std::__cxx11::basic_string<char>, name_trait_list<int, std::__cxx11::basic_string<char>, double>, double (&)[2], double>' requested here
cpp_quick/tpl_check_errors.cc:104:29: note: in instantiation of template class 'is_all_good<int, std::__cxx11::basic_string<char>, name_trait_list<int, std::__cxx11::basic_string<char>, double>, double (&)[2], double>' requested here
    typename std::enable_if<is_all_good<Args...>::value>::type>
                            ^
cpp_quick/tpl_check_errors.cc:105:6: note: in instantiation of default argument for 'my_func<int, std::__cxx11::basic_string<char>, name_trait_list<int, std::__cxx11::basic_string<char>, double>, double (&)[2], double>' required here
void my_func(Args&&... args) {
     ^~~~~~~~~~~~~~~~~~~~~~~~~
cpp_quick/tpl_check_errors.cc:115:5: note: while substituting deduced template arguments into function template 'my_func' [with Args = <int, std::__cxx11::basic_string<char>, name_trait_list<int, std::__cxx11::basic_string<char>, double>, double (&)[2], double>, Cond = (no value)]
    my_func(int{},
    ^
cpp_quick/tpl_check_errors.cc:115:5: error: no matching function for call to 'my_func'
    my_func(int{},
    ^~~~~~~
cpp_quick/tpl_check_errors.cc:105:6: note: candidate template ignored: substitution failure [with Args = <int, std::__cxx11::basic_string<char>, name_trait_list<int, std::__cxx11::basic_string<char>, double>, double (&)[2], double>]
void my_func(Args&&... args) {
     ^
*/

template<typename T, typename... Args>
struct is_all_good {
    static constexpr bool value =
        is_all_good<T>::value ?
            is_all_good<Args...>::value : false;
};
// Ensure that we decay the type
// Use is_good_custom to demonstrate different failure mechanisms and
// their respective visibility
template<typename T>
struct is_all_good<T>
    // // Unfriendly
    : is_good<typename std::remove_reference<T>::type>
    // Friendly
    // : AssertionChecker<is_good<typename std::remove_reference<T>::type>>
{ };

// Ensure that name_trait_list is unfolded
template<typename... Args>
struct is_good<name_trait_list<Args...>>
    : is_all_good<Args...> { };


template<typename... Args, typename Cond =
    typename std::enable_if<is_all_good<Args...>::value>::type>
void my_func(Args&&... args) {
    cout
        << name_trait_list<
            typename std::remove_reference<Args>::type...>::join()
        << endl;
}

int main() {
    double array[2] = {2.0, 3.0};
    // Compilable:
    my_func(int{},
        string{},
        name_trait_list<int,string,double> {},
        array,
        2.0);
    // Output if is_good<double[2]>::value == true
    //     int, std::string, name_trait_list<int, std::string, double>, double[2], double
    return 0;
}
