namespace testers {

void func();

template <typename T>
class MyClass {
public:
    T value() const;
};

extern template MyClass<int>;
extern template MyClass<float>;

}
