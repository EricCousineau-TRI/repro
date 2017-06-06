
//template <typename Key, typename Value>
//struct ReverseMap {
// public:
//  typedef std::map<Key, Value> Map;
//  struct AssignAtDeath {
//    Map& map_;
//    const Value& value_;
//    Key key_{};
//    bool has_key_{false};
//    AssignAtDeath(Map& map, const Value& value)
//        : map_(map), value_(value) {}
//    AssignAtDeath& operator=(const Key& key) {
//      has_key = true;
//      key_ = key;
//    }
//    ~AssignAtDeath() {
//      DRAKE_ASSERT(has_key_);
//      map_[key_] = value_;
//    }
//  };
//  Map& map_;
//  ReverseMap(std::map<Key, Value>& instance)
//      : map_(map) {}
//  AssignAtDeath&& operator[](const Value& value) {
//    return AssignAtDeath(map_, value);
//  }
//};

//template <typename Key, typename Value>
//struct ReverseMap<std::map<Key, Value>> : public ReverseMap<Key, Value> {};
