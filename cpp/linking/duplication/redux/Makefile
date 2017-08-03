# From: https://stackoverflow.com/questions/8623657/multiple-instances-of-singleton-across-shared-libraries-on-linux
FLAGS := $(CXXFLAGS) -fPIC
all: run
main_0: main.cpp
	$(CXX) $(FLAGS) -DMODE=0 -o $@ $< -ldl
main_1_no_rdynamic: main.cpp
	$(CXX) $(FLAGS) -DMODE=1 -o $@ $< -ldl
main_1: main.cpp
	$(CXX) $(FLAGS) -rdynamic -DMODE=1 -o $@ $< -ldl
main_2: main.cpp
	$(CXX) $(FLAGS) -DMODE=2 -o $@ $< -ldl
main_3: main.cpp
	$(CXX) $(FLAGS) -DMODE=3 -o $@ $< -ldl
main_4: main.cpp
	$(CXX) $(FLAGS) -DMODE=4 -o $@ $< -ldl
consumer_1.so: consumer.cpp
	$(CXX) $(FLAGS) -DCONSUMER='"$@"' -shared -o $@ $<
consumer_2.so: consumer.cpp
	$(CXX) $(FLAGS) -DCONSUMER='"$@"' -shared -o $@ $<
run: main_0 main_1 main_1_no_rdynamic main_2 main_3 main_4 consumer_1.so consumer_2.so
	./main_0
	./main_1_no_rdynamic
	./main_1
	./main_2
	./main_3
	./main_4
