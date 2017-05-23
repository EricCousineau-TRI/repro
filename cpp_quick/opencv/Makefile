.PHONY: all clean

all: display_image

clean:
	rm -rf *.o
	rm -rf display_image

display_image: display_image.cc
	$(CXX) $< $(shell pkg-config opencv --cflags --libs) -o $@
