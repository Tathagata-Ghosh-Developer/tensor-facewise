CXX ?= g++
NVCC ?= nvcc

TARGET_CPU := tensor_app_cpu
TARGET_CUDA := tensor_app_cuda
REPORT_ASSET_SCRIPT := scripts/generate_report_assets.py

INCLUDES := -Iinclude
CPPFLAGS := -std=c++17 -O3 -Wall -Wextra -Wpedantic $(INCLUDES)
CXXFLAGS := $(CPPFLAGS) -fopenmp
NVCCFLAGS := -std=c++17 -O3 -Xcompiler="-Wall -Wextra -Wno-pedantic -fopenmp" $(INCLUDES) -DTENSOR3D_USE_CUDA

CPU_SOURCES := src/main.cpp
CUDA_SOURCES := src/main.cpp src/cuda_kernels.cu

CUDA_HOME ?= $(shell nvcc_path=$$(command -v $(NVCC) 2>/dev/null); if [ -n "$$nvcc_path" ]; then cd "$$(dirname "$$nvcc_path")/.." && pwd; else echo /usr/local/cuda; fi)
CUDA_RPATH_DIRS := $(CUDA_HOME)/lib64 $(CUDA_HOME)/targets/x86_64-linux/lib /usr/local/cuda/lib64 /usr/local/cuda-12.6/lib64 /usr/local/cuda-12.6/targets/x86_64-linux/lib
CUDA_RPATH_FLAGS := $(foreach d,$(CUDA_RPATH_DIRS),-Xlinker -rpath -Xlinker $(d))

.PHONY: all cpu cuda report-assets clean format

all: cpu

cpu: $(TARGET_CPU)

cuda: $(TARGET_CUDA)

report-assets:
	python3 $(REPORT_ASSET_SCRIPT)

$(TARGET_CPU): $(CPU_SOURCES)
	$(CXX) $(CXXFLAGS) $(CPU_SOURCES) -o $@

$(TARGET_CUDA): $(CUDA_SOURCES)
	$(NVCC) $(NVCCFLAGS) $(CUDA_SOURCES) -o $@ -lcudart -lcublas $(CUDA_RPATH_FLAGS)

clean:
	rm -f $(TARGET_CPU) $(TARGET_CUDA)
