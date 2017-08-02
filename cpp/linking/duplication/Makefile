# From: https://stackoverflow.com/questions/8623657/multiple-instances-of-singleton-across-shared-libraries-on-linux

all: example1 example2 hello.so hello2.so

example1: main.cpp
	$(CXX) $(CXXFLAGS) -fPIC -rdynamic -o example1 main.cpp -ldl

example2: main2.cpp
	$(CXX) $(CXXFLAGS) -fPIC -rdynamic -o example2 main2.cpp -ldl

hello.so: hello.cpp
	$(CXX) $(CXXFLAGS) -fPIC -rdynamic -shared -o hello.so hello.cpp

hello2.so: hello2.cpp
	$(CXX) $(CXXFLAGS) -fPIC -rdynamic -shared -o hello2.so hello2.cpp

clean:
	rm -f example1 hello.so

.PHONY: clean