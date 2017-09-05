#include <gtest/gtest.h>

using namespace std;
using namespace ::testing;

// https://stackoverflow.com/questions/216823/whats-the-best-way-to-trim-stdstring
// trim from end (in place)
string rtrim(string s) {
  s.erase(std::find_if(s.rbegin(), s.rend(), [](int ch) {
      return !std::isspace(ch);
  }).base(), s.end());
  return s;
}

vector<string> split(const string& in, char delim = '\n') {
  vector<string> out;
  stringstream ss(in);
  string item;
  while (std::getline(ss, item, delim)) {
    out.push_back(item);
  }
  return out;
}

string join(const vector<string>& in, const string& delim) {
  ostringstream os;
  for (int i = 0; i < in.size(); ++i) {
    os << in[i];
    if (i + 1 < in.size())
      os << delim;
  }
  return os.str();
}

string indent(const string& in, const string& ind = "  ") {
  vector<string> pieces = split(in);
  return ind + join(pieces, "\n" + ind);
}

string indent(const AssertionResult& res, const string& ind = "    ") {
  return indent(res.message(), ind);
}

AssertionResult indented(const AssertionResult& res,
                         const string& ind = "    ") {
  if (!res) {
    // Do not trmi
    return AssertionFailure() << indent(res.message(), ind) << "\n";
  } else {
    return AssertionSuccess();
  }
}

AssertionResult operator||(const AssertionResult& a, const AssertionResult& b) {
  if (!a && !b) {
    return indented(AssertionFailure()
        << "\nlhs || rhs failed:\n"
        << "lhs (false):\n" << indent(a) << "\n"
        << "rhs (false):\n" << indent(b) << "\n");
  } else {
    return AssertionSuccess();
  }
}

AssertionResult operator&&(const AssertionResult& a, const AssertionResult& b) {
  if (!bool(a) || !bool(b)) {
    AssertionResult res = AssertionFailure();
    res << "\nlhs && rhs failed:\n";
    if (!bool(a)) {
      res << "lhs (false):\n" << indent(a) << "\n";
    } else {
      res << "lhs (true)\n";
    }
    if (!bool(b)) {
      res << "rhs (false):\n" << indent(b) << "\n";
    } else {
      res << "rhs (true)\n";
    }
    return indented(res);
  } else {
    return AssertionSuccess();
  }
}

GTEST_TEST(GTestOp, Stuff) {
  cout << "'" + rtrim("hello world    \n  ") + "'" << endl;
  cout << "'" + join({"a", "b"}, "+") + "'" << endl;

  AssertionResult good = AssertionSuccess();
  AssertionResult bad_a = AssertionFailure() << "A";
  AssertionResult bad_b = AssertionFailure() << "B";

  EXPECT_TRUE(good && good);
  EXPECT_TRUE(bad_a && good);
  EXPECT_TRUE(good && bad_b);
  EXPECT_TRUE(bad_a && bad_b);

  EXPECT_TRUE(good || good);
  EXPECT_TRUE(bad_a || good);
  EXPECT_TRUE(good || bad_b);
  EXPECT_TRUE(bad_a || bad_b);
}
