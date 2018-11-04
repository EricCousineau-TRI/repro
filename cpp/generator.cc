#include <cassert>
#include <iostream>
#include <functional>
#include <experimental/optional>
#include <memory>
#include <vector>

using std::cerr;
using std::endl;
using std::experimental::optional;
using std::unique_ptr;

template <typename T, typename SFINAE = void>
struct generator_value_type {
  using value_type = typename T::value_type;
};

// Enable `unique_ptr` to be an iterated type, where `nullptr` is the stopping
// criteria. Return a reference.
template<typename T>
struct generator_value_type<unique_ptr<T>> {
  using value_type = T&;
};

/**
Provides a make_generator class that can be iterated upon. Per iteration, this
returns `value_type`, inferred from `result_type`, inferred from `Func`.

See `generator_value_type<>` for how value types are extracted.

@tparam Func
  Function that returns `result_type`.
  result_type
    Must obey the following contract:
     - provides `value_type`
     - moveable
     - (optional) copy constructible
     - result_t() -> construct invalid value
     - result_t(T&&) -> construct valid value
     - operator* -> T&&
        dereference to value. First dereferences for a moveable type may
        invalidate future dereferences.
     - operator bool():
        true -> valid value, consume
        false -> invalid value, stop iterating

Partially imitates Python generator expressions:

- begin() returns an iterator that has evaluated the first expression.
- end() refers to the end of iteration (no more values to consume). Will be
  triggered on the first invalid value returned.

This makes no constraints on what `Func` can refer to.
 */
template <typename Func>
class generator {
 public:
  using result_type = decltype(std::declval<Func>()());
  using value_type = typename generator_value_type<result_type>::value_type;

  generator(Func&& func)
      : func_(std::forward<Func>(func)) {}

  template <typename OtherFunc>
  generator(generator<OtherFunc>&& other)
      : func_(std::move(other.func_)) {}

  class iterator {
   public:
    iterator() {}
    iterator(generator* parent, bool finished = false)
        : parent_(parent), finished_(finished) {
      assert(valid());
      if (!finished_) {
        ++(*this);
      }
    }
    bool valid() const {
      return parent_ != nullptr;
    }
    value_type&& operator*() {
      assert(valid());
      return std::forward<value_type>(*result_);
    }
    iterator& operator++() {
      assert(valid());
      result_ = parent_->next();
      if (!result_) {
        finished_ = true;
      }
      return *this;
    }
    bool operator==(const iterator& other) {
      // Not precise, but provides a succinct implementation to enable
      // comparison to `end`.
      return parent_ == other.parent_ && finished_ == other.finished_;
    }
    bool operator!=(const iterator& other) { return !(*this == other); }
   private:
    generator* parent_{};
    bool finished_{false};
    result_type result_;
  };

  iterator begin() { return iterator(this); }
  iterator end() { return iterator(this, true); }
  result_type next() { return func_(); }
 private:
  template <typename OtherFunc>
  friend class generator;
  Func func_;
};

template <typename result_type>
using function_generator = generator<std::function<result_type()>>;

template <typename Func>
auto make_generator(Func&& func) {
  return generator<Func>(std::forward<Func>(func));
}

template <typename result_t>
auto null_generator() {
  return make_generator([]() { return result_t{}; });
}

template <typename Generator>
class chain_func {
 public:
  using result_type = typename Generator::result_type;

  chain_func(std::vector<Generator> g_list)
      : g_list_(std::move(g_list)) {}

  result_type operator()() {
    // Advance iterator.
    if (i_ >= g_list_.size()) return {};
    if (!iter_.valid()) {
      iter_ = g().begin();
    } else {
      ++iter_;
    }
    while (iter_ == g().end()) {
      if (++i_ >= g_list_.size()) return {};
      iter_ = g().begin();
    }
    // Return value.
    return *iter_;
  }
 private:
  inline Generator& g() { return g_list_[i_]; }
  std::vector<Generator> g_list_;
  typename Generator::iterator iter_;
  int i_{0};
};

template <typename result_type>
auto make_chain(std::vector<function_generator<result_type>> list) {
  return make_generator(chain_func<function_generator<result_type>>(
      std::move(list)));
}

template <typename result_type = void, typename container>
auto make_container_generator(container&& c) {
  // Enable perfect-captures: https://stackoverflow.com/q/26831382/7829525
  struct { container value; } cc{std::forward<container>(c)};
  auto iter = std::begin(cc.value);
  using T = std::decay_t<decltype(*iter)>;
  // Infer by default, or use specified `result_type`.
  using result_type_final = std::conditional_t<
      std::is_same<result_type, void>::value, optional<T>, result_type>;
  return make_generator(
    [cc = std::move(cc), iter = std::move(iter)]()
    mutable -> result_type_final {
      if (iter != std::end(cc.value)) return *(iter++);
      return {};
    });
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
  using value_type = int;

  positive_int() {}
  positive_int(int value) : value_(value) {}
  // N.B. Seems to be *very* picky about using `move` and `T&&` here.
  int&& operator*() { return std::move(value_); }
  operator bool() const { return value_ >= 0; }
 private:
  int value_{-1};
};

class move_only_func {
 public:
  optional<int> operator()() {
    if (*value_ < 5) return (*value_)++;
    else return {};
  }
 private:
  unique_ptr<int> value_{new int{0}};
};

template <typename T>
class ref {
 public:
  using value_type = T&;

  ref() {}
  ref(T& value) : ptr_(&value) {}
  T& operator*() { return *ptr_; }
  operator bool() const { return ptr_ != nullptr; }
 private:
  T* ptr_{};
};

int main() {
  auto optional_gen = []() {
    return make_generator([i = 0]() mutable -> optional<int> {
      if (i < 10) return i++;
      return {};
    });
  };
  cerr << "simple:" << endl;
  print_container(optional_gen());

  cerr << " - breaking iteration:" << endl;
  {
    auto gen = optional_gen();
    for (int rep : {0, 1, 2}) {
      cerr << "[ break " << rep << " ] ";
      int count = 0;
      for (int value : gen) {
        cerr << value << " ";
        if (++count >= 3)
          break;
      }
      cerr << endl;
    }
    cerr << endl;
  }

  cerr << "container:" << endl;
  print_container(make_container_generator(std::vector<int>{10, 20, 30}));
  {
    const std::vector<int> c{10, 20, 30};
    print_container(make_container_generator(c));
  }

  cerr << "null:" << endl;
  print_container(null_generator<optional<int>>());

  cerr << "chain: " << endl;
  print_container(make_chain<optional<int>>({
      optional_gen(), null_generator<optional<int>>(), optional_gen()}));

  cerr << "optional<unique_ptr>:" << endl;
  auto unique = make_generator(
    [i = 0]() mutable -> optional<unique_ptr<int>> {
      if (i < 5) return unique_ptr<int>(new int{i++});
      return {};
    });
  print_container(unique);

  cerr << "positive_int:" << endl;
  print_container(make_generator(
    [i = 0]() mutable -> positive_int {
      if (i < 3) return i++;
      return -1;
    }));

  cerr << " - implicit null sequence:" << endl;
  print_container(make_generator([]() mutable { return positive_int{}; }));

  cerr << "unique_ptr:" << endl;
  print_container(make_generator([i = 0]() mutable {
    if (i < 3) return unique_ptr<int>(new int{i++});
    return unique_ptr<int>();
  }));

  cerr << "move-only function:" << endl;
  print_container(make_generator(move_only_func{}));

  {
    cerr << "references:" << endl;
    int a = 0, b = 0;
    auto ref_gen = [&]() {
      return make_generator([&]() -> ref<int> {
        if (a == 0) return ++a;
        else if (b == 0) return b += 3;
        else return {};
      });
    };
    for (auto& cur : ref_gen()) {
      auto before = cur;
      cur *= 100;
      cerr << "before: " << before << ", cur: " << cur << endl;
    }
    cerr << "a: " << a << ", b: " << b << endl;
  }

  return 0;
}
