#pragma once
#ifndef OP_TYPE_H
#define OP_TYPE_H

#include <cstdint>
#include <string>
#include <unordered_set>

namespace infini {
struct OpType {
  using underlying_t = uint16_t;
  enum : underlying_t {
    Unknown,
    Add,
    Cast,
    Clip,
    Concat,
    Div,
    Mul,
    MatMul,
    Relu,
    Sub,
    Transpose,

  } type;

  constexpr OpType(decltype(type) t) : type(t) {}
  constexpr explicit OpType(underlying_t val) : type((decltype(type))val) {}
  constexpr underlying_t underlying() const { return type; }

  bool operator==(OpType others) const { return type == others.type; }
  bool operator!=(OpType others) const { return type != others.type; }
  bool operator<(OpType others) const { return type < others.type; }

  const char *toString() const;
};

} // namespace infini

#endif // OP_TYPE_H
