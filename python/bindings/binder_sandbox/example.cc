namespace testers {

void func();

template <typename T>
class MyClass {
public:
    T value() const;
};

extern template class MyClass<int>;
extern template class MyClass<float>;

}
