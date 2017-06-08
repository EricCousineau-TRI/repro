CXX_ALL_FLAGS = -std=c++1y $(shell pkg-config --cflags --libs eigen3)
OUTPUT_DIR := build/$(CXX)

.PHONY = all dir clean bin

all: bin

dir: $(OUTPUT_DIR)

$(OUTPUT_DIR):
	mkdir $(OUTPUT_DIR)

bin: $(OUTPUT_DIR)/with_openmp $(OUTPUT_DIR)/without_openmp

$(OUTPUT_DIR)/eigen_with_openmp: eigen_sample.cc | dir
	$(CXX) -fopenmp $(CXX_ALL_FLAGS) $< -o $@

$(OUTPUT_DIR)/eigen_without_openmp: eigen_sample.cc | dir
	$(CXX) $(CXX_ALL_FLAGS) $< -o $@

$(OUTPUT_DIR)/with_openmp: simple.cc | dir
	$(CXX) -fopenmp $(CXX_ALL_FLAGS) $< -o $@

$(OUTPUT_DIR)/without_openmp: simple.cc | dir
	$(CXX) $(CXX_ALL_FLAGS) $< -o $@
