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
    iterator(generator_t* parent) : parent_(parent) {
      assert(parent_ != nullptr);
      ++(*this);
    }
    iterator() {}
    T&& operator*() {
      assert(parent_);
      return std::forward<T>(*parent_->value_);
    }
    void operator++() {
      assert(parent_);
      ++count_;
      if (!parent_->next()) {
        parent_ = nullptr;
      }
    }
    bool operator==(const iterator& other) {
      if (!other.parent_) {
        return !parent_;
      } else {
        return parent_ == other.parent_ && count_ == other.count_;
      }
    }
    bool operator!=(const iterator& other) { return !(*this == other); }
   private:
    generator_t* parent_{};
    int count_{0};
  };

  iterator begin() { return iterator{this}; }
  iterator end() { return iterator{}; }
 private:
  friend class iterator;
  bool next() {
    value_ = func_();
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
      cerr << "next " << i_ << endl;
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
auto chain(std::vector<generator_t<T>> g) {
  return generator<int>(chain_t<int>(std::move(g)));
}

int main() {
  auto make = []() {
    return generator<int>([i = 0]() mutable -> optional<int> {
      cerr << "iter " << i << endl;
      if (i < 10) return i++;
      return {};
    });
  };
  for (int value : make()) {
    cerr << "value: " << value << endl;
  }
  cerr << endl;
  auto ch = chain<int>({make(), make()});
  for (int value : ch) {
    cerr << "value: " << value << endl;
  } 
  cerr << endl;
  return 0;
}
