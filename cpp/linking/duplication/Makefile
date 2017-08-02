example1: main.cpp hello.so
	$(CXX) $(CXXFLAGS) -fPIC -rdynamic -o example1 main.cpp -ldl

hello.so: hello.cpp
	$(CXX) $(CXXFLAGS) -fPIC -rdynamic -shared -o hello.so hello.cpp

clean:
	rm -f example1 hello.so

.PHONY: clean