# Makefile for nbody simulation
args ?= -f ../src/solar_system.txt

name = nbody
src_dir = src
build_dir = build
object_dir = $(build_dir)/obj
main_obj = $(object_dir)/main.o
vector_obj = $(object_dir)/vector.o
particle_obj = $(object_dir)/particle.o
system_obj = $(object_dir)/system.o
tree_obj = $(object_dir)/tree.o
object_files = $(tree_obj) $(vector_obj) $(particle_obj) $(system_obj) $(main_obj)
module_dir = $(build_dir)/pcm
vector_pcm = $(module_dir)/vector.pcm
particle_pcm = $(module_dir)/particle.pcm
system_pcm = $(module_dir)/system.pcm
tree_pcm = $(module_dir)/tree.pcm

CCCL_ROOT ?= $(CURDIR)/external/cccl
PROJECT_CPPFLAGS = -I$(CURDIR) -I$(CURDIR)/include
CCCL_CPPFLAGS = -I$(CCCL_ROOT)/libcudacxx/include -I$(CCCL_ROOT)/thrust -I$(CCCL_ROOT)/cub -I$(CCCL_ROOT)/cudax/include

UNAME_S := $(shell uname -s)

ifeq ($(UNAME_S),Darwin)
	CXX := /opt/homebrew/opt/llvm@18/bin/clang++
	SDKROOT ?= $(shell xcrun --show-sdk-path)
	OMP_PREFIX ?= /opt/homebrew/opt/libomp
	CXXFLAGS = -std=c++23 -stdlib=libc++ -fopenmp -I$(OMP_PREFIX)/include $(PROJECT_CPPFLAGS) $(CCCL_CPPFLAGS) -isysroot $(SDKROOT)
	LDFLAGS = -stdlib=libc++ -L$(OMP_PREFIX)/lib -Wl,-rpath,$(OMP_PREFIX)/lib -fopenmp
	METAL_FLAGS = -framework Metal -framework MetalKit -framework Cocoa -framework QuartzCore
	
	vector_import = -fmodule-file=vector=$(vector_pcm)
	particle_imports = $(vector_import)
	system_imports = $(vector_import) -fmodule-file=particle=$(particle_pcm)
	tree_imports = $(system_imports) -fmodule-file=system=$(system_pcm)
	main_imports = $(tree_imports)
else
	CXX ?= g++
	CXXFLAGS = -std=c++23 -fmodules-ts -fopenmp $(PROJECT_CPPFLAGS) $(CCCL_CPPFLAGS)
	LDFLAGS = -fopenmp
endif

.PHONY: all run clean test test-cuda viewer

all : $(build_dir)/$(name)

$(build_dir)/$(name) : $(build_dir) $(object_files)
	$(CXX) $(object_files) $(LDFLAGS) -o $@

$(build_dir) :
	mkdir -p $(build_dir)

$(object_dir) :
	mkdir -p $(object_dir)

ifeq ($(UNAME_S),Darwin)
$(module_dir) :
	mkdir -p $(module_dir)

$(vector_obj): $(src_dir)/vector.cppm | $(object_dir) $(module_dir)
	$(CXX) $(CXXFLAGS) -c $< -o $@ -fmodule-output=$(vector_pcm)

$(particle_obj): $(src_dir)/particle.cppm $(vector_obj) | $(object_dir) $(module_dir)
	$(CXX) $(CXXFLAGS) -c $< -o $@ -fmodule-output=$(particle_pcm) $(particle_imports)

$(system_obj): $(src_dir)/system.cppm $(particle_obj) $(vector_obj) | $(object_dir) $(module_dir)
	$(CXX) $(CXXFLAGS) -c $< -o $@ -fmodule-output=$(system_pcm) $(system_imports)

$(tree_obj): $(src_dir)/tree.cppm $(particle_obj) $(vector_obj) $(system_obj) | $(object_dir) $(module_dir)
	$(CXX) $(CXXFLAGS) -c $< -o $@ -fmodule-output=$(tree_pcm) $(tree_imports)

$(main_obj): $(src_dir)/main.cpp $(system_obj) $(particle_obj) $(vector_obj) | $(object_dir) $(module_dir)
	$(CXX) $(CXXFLAGS) -c $< -o $@ $(main_imports)

viewer: $(build_dir)/nbody_viewer
	./$(build_dir)/nbody_viewer

$(build_dir)/nbody_viewer: apps/viewer/main.mm | $(build_dir)
	$(CXX) $(CXXFLAGS) $< $(LDFLAGS) $(METAL_FLAGS) -o $@

else
# Basic non-darwin rules
$(vector_obj): $(src_dir)/vector.cppm | $(object_dir)
	$(CXX) $(CXXFLAGS) -x c++ -c $< -o $@

$(particle_obj): $(src_dir)/particle.cppm $(vector_obj) | $(object_dir)
	$(CXX) $(CXXFLAGS) -x c++ -c $< -o $@
endif

CMAKE ?= cmake
CTEST ?= ctest
CMAKE_GENERATOR ?= Ninja

test:
	$(CMAKE) -S . -B build/cpu-tests -G "$(CMAKE_GENERATOR)" -DBUILD_TESTING=ON -DNBODY_ENABLE_CUDA=OFF -DNBODY_BUILD_VIEWER=OFF -DCMAKE_CXX_COMPILER="$(CXX)"
	$(CMAKE) --build build/cpu-tests --parallel
	$(CTEST) --test-dir build/cpu-tests --output-on-failure

test-cuda:
	$(CMAKE) -S . -B build/cuda-tests -G "$(CMAKE_GENERATOR)" -DBUILD_TESTING=ON -DNBODY_ENABLE_CUDA=ON -DNBODY_BUILD_VIEWER=OFF
	$(CMAKE) --build build/cuda-tests --parallel
	$(CTEST) --test-dir build/cuda-tests -L cuda --output-on-failure

clean :
	rm -rf $(build_dir)
	rm -rf gcm.cache
	rm -rf .cache/clangmodules
