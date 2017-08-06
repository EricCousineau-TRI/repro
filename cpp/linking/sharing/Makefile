# From: https://stackoverflow.com/questions/8623657/multiple-instances-of-singleton-across-shared-libraries-on-linux
FLAGS := $(CXXFLAGS)
all: run
main_0: main.cpp
	$(CXX) $(FLAGS) -o $@ $< -ldl
consumer_1.so: consumer.cpp
	$(CXX) $(FLAGS) -DCONSUMER='"$@"' -fPIC -shared -o $@ $<
consumer_2.so: consumer.cpp
	$(CXX) $(FLAGS) -DCONSUMER='"$@"' -fPIC -shared -o $@ $<
run: main_0 consumer_1.so consumer_2.so
	./main_0
