// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef MSCCLPP_SERIALIZATION_HPP_
#define MSCCLPP_SERIALIZATION_HPP_

#include <algorithm>
#include <string>
#include <type_traits>
#include <vector>

namespace mscclpp::detail {

template <typename T>
void serialize(std::vector<char>& buffer, const T& value) {
  static_assert(std::is_trivially_copyable_v<T>, "Can only serialize trivially copyable types");
  const char* data = reinterpret_cast<const char*>(&value);
  std::copy_n(data, sizeof(T), std::back_inserter(buffer));
}

template<typename CharT, typename Traits, typename Allocator>
void serialize(std::vector<char>& buffer, const std::basic_string<CharT, Traits, Allocator>& str) {
  serialize(buffer, static_cast<std::size_t>(str.size()));
  const char* data = reinterpret_cast<const char*>(str.data());
  std::copy_n(data, str.size() * sizeof(CharT), std::back_inserter(buffer));
}

template <typename T>
void serialize(std::vector<char>& buffer, const std::vector<T>& values) {
  serialize(buffer, static_cast<std::size_t>(values.size()));
  for (const T& value : values) {
    serialize(buffer, value);
  }
}

template <typename T>
std::vector<char>::const_iterator deserialize(const std::vector<char>::const_iterator& pos, T& value) {
  static_assert(std::is_trivially_copyable_v<T>, "Can only deserialize trivially copyable types");
  std::copy_n(pos, sizeof(T), reinterpret_cast<char*>(&value));
  return pos + sizeof(T);
}
template<typename CharT, typename Traits, typename Allocator>
std::vector<char>::const_iterator deserialize(const std::vector<char>::const_iterator& pos, std::basic_string<CharT, Traits, Allocator>& str) {
  std::size_t size;
  auto it = deserialize(pos, size);
  str.resize(size);
  std::copy_n(it, size * sizeof(CharT), reinterpret_cast<char*>(str.data()));
  return it + size * sizeof(CharT);
}

template <typename T>
std::vector<char>::const_iterator deserialize(const std::vector<char>::const_iterator& pos, std::vector<T>& values) {
  std::size_t size;
  auto it = deserialize(pos, size);
  values.resize(size);
  for (std::size_t i = 0; i < size; ++i) {
    it = deserialize(it, values[i]);
  }
  return it;
}

}  // namespace mscclpp::detail

#endif  // MSCCLPP_SERIALIZATION_HPP_
