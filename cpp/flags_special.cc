// Purpose: Test "faking" out flags.
#include <iostream>
#include <sstream>
#include <string>

using std::string;
using std::cout;
using std::endl;

enum FlagA : int {
  kFlagANone = 0,
  kFlagA1 = 1 << 0,
  kFlagA2 = 1 << 1,
  kFlagA3 = 1 << 2,
};

struct FlagB {
  string name;
};

bool operator==(const FlagB& lhs, const FlagB& rhs) {
  return lhs.name == rhs.name;
}
bool operator!=(const FlagB& lhs, const FlagB& rhs) {
  return lhs.name != rhs.name;
}

const FlagB
  kFlagBNone{"None"},
  kFlagBInherit{"Inherit"},
  kFlagB1{"1"},
  kFlagB2{"2"};

class Flags {
 public:
  Flags(const int& flag)
    : flags_a_(static_cast<FlagA>(flag)) {}

  Flags(const FlagB& flag)
    : flag_b_(flag) {
    // Explicit prevent kFlagBNone from being used.
    if (flag == kFlagBNone) {
      throw std::runtime_error("Cannot explicit pass kFlagBNone");
    }
  }

  Flags& operator|=(const Flags& other) {
    flags_a_ = static_cast<FlagA>(flags_a_ | other.flags_a_);
    if (flag_b_ == kFlagBNone) {
      flag_b_ = other.flag_b_;
    } else {
      throw std::runtime_error(
          "Cannot have multiple FlagB flags. "
          "Can only do operator| iff (flags.b() == kFlagBNone) == true.");
    }
    return *this;
  }

  Flags operator|(const Flags& rhs) const {
    return Flags(*this) |= rhs;
  }

  bool operator&(const Flags& rhs) const {
    if (flags_a_ & rhs.flags_a_) {
      return flag_b_ == rhs.flag_b_;
    } else {
      return false;
    }
  }

  operator string() const {
    std::ostringstream os;
    os << "(";
    if (flags_a_ & kFlagA1)
      os << "kFlagA1 | ";
    if (flags_a_ & kFlagA2)
      os << "kFlagA2 | ";
    if (flags_a_ & kFlagA3)
      os << "kFlagA3 | ";
    if (flag_b_ != kFlagBNone)
      os << "kFlagB::" << flag_b_.name << " | ";
    os << ")";
    return os.str();
  }

 private:
  // Allow these to accumulate.
  FlagA flags_a_{kFlagANone};
  // These cannot accumulate.
  FlagB flag_b_{kFlagBNone};
};

std::ostream& operator<<(std::ostream& os, const Flags& flags) {
  return os << string(flags);
}

// Make | compatible for FlagA + FlagB.
Flags operator|(const int& lhs, const FlagB& rhs) {
  return Flags(lhs) | rhs;
}
// Use implicit conversion since there is no ambiguity with FlagA.
Flags operator|(const FlagB& lhs, const Flags& rhs) {
  return Flags(lhs) | rhs;
}

#define EVAL(x) std::cout << ">>> " #x ";" << std::endl; x; std::cout << std::endl
#define PRINT(x) ">>> " #x << std::endl << (x) << std::endl << std::endl

int main() {
  EVAL(Flags flags = kFlagA1 | kFlagA2 | kFlagB1);
  cout
    << PRINT(flags)
    << PRINT(flags & kFlagA1)
    << PRINT(flags & kFlagA2)
    << PRINT(flags & kFlagA3)
    << PRINT(flags & kFlagB1)
    << PRINT(flags & kFlagB2)
    << PRINT(flags & kFlagBInherit);
  return 0;
}
