#include <algorithm>
#include <list>
#include <fmt/format.h>

struct Item {
  int stamp{};
  int id{};

  bool operator==(const Item& other) const {
    return stamp == other.stamp && id == other.id;
  }
  bool operator!=(const Item& other) const {
    return *this != other;
  }
};

int main() {
  std::vector<Item> items{
    Item{.stamp=0, .id=0},
    Item{.stamp=1, .id=1},
    Item{.stamp=1, .id=2},
    Item{.stamp=2, .id=3}};

  auto print_items = [&items]() {
    fmt::print("{{\n");
    for (auto& item : items) {
      fmt::print("  Item{{.stamp={}, .id={}}},\n", item.stamp, item.id);
    }
    fmt::print("}}\n");
  };

  auto insert = [&items](const Item& item) {
    auto predicate = [item](const Item& other) {
      return other.stamp <= item.stamp;
    };
    auto iter = std::find_if(items.begin(), items.end(), predicate);
    const int index = iter - items.begin();
    fmt::print("index: {}\n", index);
    items.insert(iter, item);
  };

  insert(Item{.stamp=2, .id=4});
  insert(Item{.stamp=0, .id=5});
  insert(Item{.stamp=10, .id=6});

  print_items();

  return 0;
}
