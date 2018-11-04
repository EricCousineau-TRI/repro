#include <cassert>
#include <iostream>
#include <functional>
#include <experimental/optional>
#include <memory>
#include <vector>

using std::cerr;
using std::endl;
using std::experimental::optional;
using std::experimental::nullopt;
using std::unique_ptr;

/**
Provides a generator class that can be iterated upon:
@tparam T
  Type to return when iterating
@tparam wrapper (default: optional<T>)
  Class that must follow this contract:
   - moveable
   - (optional) copy constructible
   - wrapper() -> construct invalid value
   - wrapper(T&&) -> construct valid value
   - operator* -> T&&
      dereference to value, moveable
   - operator bool():
      true -> valid value, consume
      false -> invalid value, stop iterating
Imitates Python generator expressions:
- begin() returns an iterator that has evaluated the first expression
- end() refers to the end of iteration (no more values to consume)
- Can only be iterated once; afterwards, it will always return
  `begin() == end()`.
 */
template <typename T, typename wrapper = optional<T>>
class generator_t {
 public:
  using Func = std::function<wrapper()>;
  generator_t(Func func) : func_(func) {}

  class iterator {
   public:
    iterator() {}
    iterator(generator_t* parent, bool is_end) : parent_(parent) {
      assert(valid());
      if (!is_end) {
        ++(*this);
      }
    }
    bool valid() const {
      return parent_ != nullptr;
    }
    T&& operator*() {
      assert(valid());
      return std::forward<T>(*parent_->value_);
    }
    void operator++() {
      assert(valid());
      ++count_;
      if (!parent_->eval_next()) {
        count_ = 0;
      }
    }
    bool operator==(const iterator& other) {
      return parent_ == other.parent_ && count_ == other.count_;
    }
    bool operator!=(const iterator& other) { return !(*this == other); }
   private:
    generator_t* parent_{};
    int count_{0};
  };

  iterator begin() { return iterator(this, false); }
  iterator end() { return iterator(this, true); }

  // May invalidate iterator state; can restore iterator state with `++iter`.
  wrapper next() {
    eval_next();
    return std::move(value_);
  }
 private:
  friend class iterator;
  bool eval_next() {
    if (func_) {
      value_ = func_();
    }
    if (!value_) {
      func_ = {};
    }
    return bool{value_};
  }
  Func func_;
  wrapper value_;
};

template <typename T, typename wrapper = optional<T>>
auto generator(const typename generator_t<T, wrapper>::Func& func) {
  return generator_t<T, wrapper>(func);
}

template <typename T, typename wrapper>
class chain_func {
 public:
  chain_func(std::vector<generator_t<T, wrapper>> g_list)
      : g_list_(std::move(g_list)) {}

  wrapper operator()() {
    if (!next()) return {};
    return *iter_;
  }

 private:
  inline generator_t<T>& g() { return g_list_[i_]; }

  inline bool next() {
    if (i_ >= g_list_.size()) return false;
    if (!iter_.valid()) {
      iter_ = g().begin();
    } else {
      ++iter_;
    }
    while (iter_ == g().end()) {
      if (++i_ >= g_list_.size()) return false;
      iter_ = g().begin();
    }
    return true;
  }

  std::vector<generator_t<T>> g_list_;
  typename generator_t<T>::iterator iter_;
  int i_{0};
};

template <typename T, typename wrapper = optional<T>>
auto chain(std::vector<generator_t<T, wrapper>> g) {
  return generator<int, wrapper>(chain_func<int, wrapper>(std::move(g)));
}

template <typename T>
std::ostream& operator<<(std::ostream& os, const unique_ptr<T>& p) {
  if (p) os << "unique_ptr(" << *p << ")";
  else os << "unique_ptr(nullptr)";
  return os;
}

template <typename Container>
void print_container(Container&& gen) {
  for (int rep : {0, 1}) {
    cerr << "[ " << rep << " ] ";
    for (auto&& value : gen) {
      cerr << value << " ";
    }
    cerr << endl;
  }
  cerr << endl;
}

class positive_int {
 public:
  positive_int() {}
  positive_int(int value) : value_(value) {}
  // N.B. Seems to be *very* picky about using `move` and `T&&` here.
  int&& operator*() { return std::move(value_); }
  operator bool() const { return value_ >= 0; }
 private:
  int value_{-1};
};

int main() {
  auto simple_gen = []() {
    return generator<int>([i = 0]() mutable -> optional<int> {
      if (i < 10) return i++;
      return {};
    });
  };
  cerr << "simple:" << endl;
  print_container(simple_gen());
  cerr << "chain:" << endl;
  print_container(chain<int>({simple_gen(), generator<int>({}), simple_gen()}));
  cerr << "optional<unique_ptr>:" << endl;
  auto unique = generator<unique_ptr<int>>(
    [i = 0]() mutable -> optional<unique_ptr<int>> {
      if (i < 5) return unique_ptr<int>(new int{i++});
      return {};
    });
  print_container(unique);
  cerr << "positive_int:" << endl;
  print_container(generator<int, positive_int>(
    [i = 0]() mutable {
      if (i < 3) return i++;
      return -1;
    }));
  cerr << "unique_ptr:" << endl;
  print_container(generator<int, unique_ptr<int>>([i = 0]() mutable {
    if (i < 3) return unique_ptr<int>(new int{i++});
    return unique_ptr<int>();
  }));
  return 0;
}
