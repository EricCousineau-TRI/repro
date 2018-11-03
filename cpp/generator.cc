#include <cassert>
#include <iostream>
#include <experimental/optional>

using std::experimental::optional;

template <typename Return>
class value {
 public:
  void operator=(Return value) {
    value_ = std::forward<Return>(value);
  }
  Return&& operator*() { return std::forward<Return>(*value_); }
 private:
  optional<Return> value_{};
};

template <typename Func>
class impl {
 public:
  using Return = decltype(std::declval<Func>()(std::declval<bool&>()));
  impl(Func&& func) : func_(std::forward<Func>(func)) {}

  struct end_t {};
  class iterator {
   public:
    iterator(impl* parent) : parent_(parent) {}
    iterator() : parent_() {}
    Return&& operator*() { return *parent_->value_; }
    void operator++() { parent_->next(); }
    bool operator!=(const iterator& other) {
      assert(other.parent_ == nullptr);
      return !parent_->finished_;
    }
   private:
    impl* parent_{};
  };

  iterator begin() { return iterator{this}; }
  iterator end() { return iterator{}; }

 private:
  friend class iterator;
  void next() {
    assert(!finished_);
    bool keep = false;
    value_ = func_(keep);
    finished_ = !keep;
  }

  Func func_;
  value<Return> value_;
  bool initialized_{};
  bool finished_{};
};

template <typename Func>
auto generator(Func&& func) {
  return impl<Func>(std::forward<Func>(func));
}


int main() {
  auto gen = generator([i = 0](bool& keep) mutable {
    keep = i < 10;
    return ++i;
  });
  for (int& value : gen) {
    std::cout << value << " ";
  }
  std::cout << std::endl;
  return 0;
}
