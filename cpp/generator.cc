#include <cassert>
#include <iostream>
#include <functional>
#include <experimental/optional>
#include <vector>

using std::cerr;
using std::endl;
using std::experimental::optional;
using std::experimental::nullopt;

template <typename T>
class generator_t {
 public:
  using Func = std::function<optional<T>()>;
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
      if (!parent_->next()) {
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
 private:
  friend class iterator;
  bool next() {
    if (func_) {
      value_ = func_();
    }
    if (!value_) {
      func_ = {};
    }
    return bool{value_};
  }
  Func func_;
  optional<T> value_;
};

template <typename T>
auto generator(const typename generator_t<T>::Func& func) {
  return generator_t<T>(func);
}

template <typename T>
class chain_func {
 public:
  chain_func(std::vector<generator_t<T>> g_list): g_list_(std::move(g_list)) {}

  optional<T> operator()() {
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

template <typename T>
auto chain(std::vector<generator_t<T>> g) {
  return generator<int>(chain_func<int>(std::move(g)));
}

template <typename T>
void print_generator(generator_t<T>&& gen) {
  for (int rep : {0, 1}) {
    cerr << "[ " << rep << " ] ";
    for (int value : gen) {
      cerr << value << " ";
    }
    cerr << endl;
  }
  cerr << endl;
}

int main() {
  auto simple_gen = []() {
    return generator<int>([i = 0]() mutable -> optional<int> {
      if (i < 10) return i++;
      return {};
    });
  };
  cerr << "simple:" << endl;
  print_generator(simple_gen());
  cerr << "chain:" << endl;
  print_generator(chain<int>({simple_gen(), generator<int>({}), simple_gen()}));
  return 0;
}
