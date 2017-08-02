example1: main.cpp hello.so hello2.so
	$(CXX) $(CXXFLAGS) -fPIC -rdynamic -o example1 main.cpp -ldl

hello.so: hello.cpp
	$(CXX) $(CXXFLAGS) -fPIC -rdynamic -shared -o hello.so hello.cpp

hello2.so: hello2.cpp
	$(CXX) $(CXXFLAGS) -fPIC -rdynamic -shared -o hello2.so hello2.cpp

clean:
	rm -f example1 hello.so

.PHONY: clean