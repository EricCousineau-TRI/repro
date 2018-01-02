#pragma once

// BEGIN: drake/common/hash.h
#include <cassert>
#include <cstddef>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <typeindex>
#include <utility>
#include <vector>

#include "cpp/type_pack.h"

namespace simple_converter {

// Simple (less robust) version of Drake's SystemScalarConverter
template <
    template <typename...> class Tpl,
    template <typename...> class Ptr = std::unique_ptr>
class SimpleConverter {
 public:
  typedef std::function<void*(const void*)> ErasedConverter;
  typedef std::pair<size_t, size_t> Key;  
  typedef std::map<Key, ErasedConverter> Conversions;

  SimpleConverter() = default;

  template <typename To, typename From>
  using Converter = std::function<Ptr<To> (const From&)>;

  template <typename To, typename From>
  inline static Key get_key() {
    return Key(type_hash<To>(), type_hash<From>());
  }

  template <typename To, typename From>
  void Add(const Converter<To, From>& converter) {
    ErasedConverter erased = [converter](const void* from_raw) {
      const From* from = static_cast<const From*>(from_raw);
      return new Ptr<To>(converter(*from));
    };
    Key key = get_key<To, From>();
    AddErased(key, erased);
  }

  template <typename To, typename From>
  void AddCopyConverter() {
    Converter<To, From> converter = [](const From& from) {
      return Ptr<To>(new To(from));
    };
    Add(converter);
  }

  template <typename To, typename From>
  Ptr<To> Convert(const From& from) {
    Key key = get_key<To, From>();
    // Should not attempt idempotent conversion.
    assert(key.first != key.second);
    auto iter = conversions_.find(key);
    assert(iter != conversions_.end());
    ErasedConverter erased = iter->second;
    Ptr<To>* tmp = static_cast<Ptr<To>*>(erased(&from));
    assert(tmp != nullptr);
    Ptr<To> out(std::move(*tmp));
    delete tmp;
    return out;
  }

 private:
  Conversions conversions_;

  void AddErased(Key key, ErasedConverter erased) {
    assert(conversions_.find(key) == conversions_.end());
    conversions_[key] = erased;
  }
};

}  // namespace simple_converter
