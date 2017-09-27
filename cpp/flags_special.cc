// Purpose: Test "faking" out flags.
#include <iostream>
#include <sstream>
#include <string>

using std::string;
using std::cout;
using std::endl;

enum Field : int {
  kFieldNone = 0,
  kField1 = 1 << 0,
  kField2 = 1 << 1,
  kField3 = 1 << 2,
};

struct Descriptor {
  string name;
};

bool operator==(const Descriptor& lhs, const Descriptor& rhs) {
  return lhs.name == rhs.name;
}
bool operator!=(const Descriptor& lhs, const Descriptor& rhs) {
  return lhs.name != rhs.name;
}

#define EVAL(x) std::cout << ">>> " #x ";" << std::endl; x; std::cout << std::endl
#define PRINT(x) ">>> " #x << std::endl << (x) << std::endl << std::endl

class Flags;
std::ostream& operator<<(std::ostream& os, const Flags& flags);

const Descriptor
  kDescriptorNone{"None"},
  kDescriptorInherit{"Inherit"},
  kDescriptor1{"1"},
  kDescriptor2{"2"};

class Flags {
 public:
  Flags(const int& flag)
    : flags_a_(static_cast<Field>(flag)) {}

  Flags(const Descriptor& flag)
    : flag_b_(flag) {
    // Explicit prevent kDescriptorNone from being used.
    if (flag == kDescriptorNone) {
      throw std::runtime_error("Cannot explicit pass kDescriptorNone");
    }
  }

  Flags& operator|=(const Flags& other) {
    flags_a_ = static_cast<Field>(flags_a_ | other.flags_a_);
    if (flag_b_ == kDescriptorNone) {
      flag_b_ = other.flag_b_;
    } else {
      throw std::runtime_error(
          "Cannot have multiple Descriptor flags. "
          "Can only do operator| iff (flags.b() == kDescriptorNone) == true.");
    }
    return *this;
  }

  Flags operator|(const Flags& rhs) const {
    return Flags(*this) |= rhs;
  }

  bool operator&(const Flags& rhs) const {
    if (flags_a_ & rhs.flags_a_) {
      return true;
    } else {
      return flag_b_ == rhs.flag_b_;
    }
  }

  operator string() const {
    std::ostringstream os;
    os << "(";
    if (flags_a_ & kField1)
      os << "kField1 | ";
    if (flags_a_ & kField2)
      os << "kField2 | ";
    if (flags_a_ & kField3)
      os << "kField3 | ";
    if (flag_b_ != kDescriptorNone)
      os << "kDescriptor::" << flag_b_.name << " | ";
    os << ")";
    return os.str();
  }

 private:
  // Allow these to accumulate.
  Field flags_a_{kFieldNone};
  // These cannot accumulate.
  Descriptor flag_b_{kDescriptorNone};
};

std::ostream& operator<<(std::ostream& os, const Flags& flags) {
  return os << string(flags);
}

// Make | compatible for Field + Descriptor.
Flags operator|(const int& lhs, const Descriptor& rhs) {
  return Flags(lhs) | rhs;
}
// Use implicit conversion since there is no ambiguity with Field.
Flags operator|(const Descriptor& lhs, const Flags& rhs) {
  return Flags(lhs) | rhs;
}

int main() {
  EVAL(Flags flags = kField1 | kField2 | kDescriptor1);
  cout
    << PRINT(flags)
    << PRINT(flags & kField1)
    << PRINT(flags & kField2)
    << PRINT(flags & kField3)
    << PRINT(flags & kDescriptor1)
    << PRINT(flags & kDescriptor2)
    << PRINT(flags & kDescriptorInherit);
  return 0;
}
