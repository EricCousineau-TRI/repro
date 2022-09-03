#include <string>

std::string Rlocation(const std::string& respath);

typedef int (*func_t)();

func_t LoadSymbol(const std::string& path, const std::string& func_name);
