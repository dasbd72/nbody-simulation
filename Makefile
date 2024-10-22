CXX := g++
CXXFLAGS := -std=c++11 -O3
NVFLAGS := $(CXXFLAGS) -Xptxas=-v -arch=sm_61
TARGET := nbody

.PHONY: all
all: $(TARGET)

.PHONY: nbody
nbody: nbody.cu
	nvcc $(NVFLAGS) -o nbody nbody.cu

.PHONY: clean
clean:
	rm -f $(TARGET)
