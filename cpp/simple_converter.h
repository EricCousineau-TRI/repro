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

template <template <typename...> class Tpl>
class SimpleConverterAttorney;

// Simple (less robust) version of Drake's SystemScalarConverter
template <template <typename...> class Tpl>
class SimpleConverter {
 public:
  typedef std::function<void*(const void*)> ErasedConverter;
  typedef std::pair<size_t, size_t> Key;  
  typedef std::map<Key, ErasedConverter> Conversions;

  SimpleConverter() = default;

  template <typename To, typename From>
  using Converter = std::function<std::unique_ptr<To> (const From&)>;

  template <typename To, typename From>
  inline static Key get_key() {
    return Key(type_hash<To>(), type_hash<From>());
  }

  template <typename To, typename From>
  void Add(const Converter<To, From>& converter) {
    ErasedConverter erased = [converter](const void* from_raw) {
      const From* from = static_cast<const From*>(from_raw);
      return converter(*from).release();
    };
    Key key = get_key<To, From>();
    AddErased(key, erased);
  }

  template <typename To, typename From>
  void AddCopyConverter() {
    Converter<To, From> converter = [](const From& from) {
      return std::unique_ptr<To>(new To(from));
    };
    Add(converter);
  }

  template <typename To, typename From>
  std::unique_ptr<To> Convert(const From& from) {
    Key key = get_key<To, From>();
    // Should not attempt idempotent conversion.
    assert(key.first != key.second);
    auto iter = conversions_.find(key);
    assert(iter != conversions_.end());
    ErasedConverter erased = iter->second;
    To* out = static_cast<To*>(erased(&from));
    assert(out != nullptr);
    return std::unique_ptr<To>(out);
  }

 private:
  Conversions conversions_;

  void AddErased(Key key, ErasedConverter erased) {
    assert(conversions_.find(key) == conversions_.end());
    conversions_[key] = erased;
  }
};

}  // namespace simple_converter
