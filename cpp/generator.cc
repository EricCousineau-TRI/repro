#include <cassert>
#include <iostream>
#include <functional>
#include <experimental/optional>
#include <vector>

using std::experimental::optional;
using std::experimental::nullopt;

template <typename T>
class generator_t {
 public:
  using Func = std::function<optional<T>()>;
  generator_t(Func func) : func_(func) {}

  class iterator {
   public:
    iterator(generator_t* parent) : parent_(parent) { parent_->next(); }
    iterator() {}
    T&& operator*() { return std::forward<T>(*parent_->value_); }
    void operator++() { parent_->next(); }
    bool operator==(const iterator& other) {
      if (!other.parent_) {
        return !parent_ || parent_->finished_;
      } else {
        return this == &other;
      }
    }
    bool operator!=(const iterator& other) { return !(*this == other); }
   private:
    generator_t* parent_{};
  };

  iterator begin() { return iterator{this}; }
  iterator end() { return iterator{}; }
 private:
  friend class iterator;
  void next() {
    value_ = func_();
    if (!value_) finished_ = true;
  }
  Func func_;
  optional<T> value_;
  bool finished_{};
};

template <typename T>
auto chain(const typename generator_t<T>::Func& func) {
  return generator_t<T>(func);
}

template <typename T>
class chain_t {
 public:
  chain_t(std::vector<generator_t<T>> g) : g_(std::move(g)) {
    if (g_.size() > 0) {
      iter_ = g_[0].begin();
    }
  }
  optional<T> operator()() {
    if (i_ >= g_.size()) return {};
    while (iter_ == g_[i_].end()) {
      if (++i_ >= g_.size()) return {};
      iter_ = g_[i_].begin();
    }
    optional<T> value = *iter_;
    ++iter_;
    return std::move(value);
  }
 private:
  std::vector<generator_t<T>> g_;
  typename generator_t<T>::iterator iter_;
  int i_{0};
};

template <typename T>
auto make_chain(std::vector<generator_t<T>> g) {
  return chain<int>(chain_t<int>(std::move(g)));
}

int main() {
  auto make = []() {
    return chain<int>([i = 0]() mutable -> optional<int> {
      if (i < 10) return i++;
      return {};
    });
  };
  for (int value : make()) {
    std::cerr << value << " ";
  }
  std::cerr << std::endl;
  auto ch = make_chain<int>({make(), make()});
  for (int value : ch) {
    std::cerr << value << " ";
  } 
  std::cerr << std::endl;
  return 0;
}
