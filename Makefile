# Makefile for nbody simulation
args ?= -f ../src/solar_system.txt
test_args ?= -f src/solar_system.txt

name = nbody
CXX = g++
CXXFLAGS = -std=c++20 -fmodules-ts -fopenmp
LDFLAGS = -fopenmp

src_dir = src

build_dir = build
object_dir = $(build_dir)/obj
main_obj = $(object_dir)/main.o
vector_obj = $(object_dir)/vector.o
particle_obj = $(object_dir)/particle.o
system_obj = $(object_dir)/system.o
object_files = $(vector_obj) $(particle_obj) $(system_obj) $(main_obj)

.PHONY: all run test-run clean debug debug_run

all : $(build_dir)/$(name)

$(build_dir)/$(name) : $(build_dir) $(object_files)
	$(CXX) $(object_files) $(LDFLAGS) -o $(build_dir)/$(name)

$(build_dir) :
	mkdir $(build_dir)

$(object_dir) :
	mkdir $(object_dir)

$(vector_obj): $(src_dir)/vector.ixx | $(object_dir)
	$(CXX) $(CXXFLAGS) -x c++ -c $< -o $@

$(particle_obj): $(src_dir)/particle.ixx $(vector_obj) | $(object_dir)
	$(CXX) $(CXXFLAGS) -x c++ -c $< -o $@

$(system_obj): $(src_dir)/system.ixx $(particle_obj) $(vector_obj) | $(object_dir)
	$(CXX) $(CXXFLAGS) -x c++ -c $< -o $@

$(main_obj): $(src_dir)/main.cpp $(system_obj) $(particle_obj) $(vector_obj) | $(object_dir)
	$(CXX) $(CXXFLAGS) -c $< -o $@

run : $(build_dir)/$(name)
	cd $(build_dir) &&\
	./$(name) $(args)
	cd src &&\
	python3 plt_trajectory.py 

test-run : $(build_dir)/$(name)
	./$(build_dir)/$(name) $(test_args)

# Debug rules
debug_name = $(name)_debug
debug_build_dir = $(build_dir)/debug
debug_object_dir = $(debug_build_dir)/obj
debug_main_obj = $(debug_object_dir)/main.o
debug_vector_obj = $(debug_object_dir)/vector.o
debug_particle_obj = $(debug_object_dir)/particle.o
debug_system_obj = $(debug_object_dir)/system.o
debug_object_files = $(debug_vector_obj) $(debug_particle_obj) $(debug_system_obj) $(debug_main_obj)
DEBUG_CXXFLAGS = $(CXXFLAGS) -g

debug : $(debug_build_dir)/$(debug_name)

$(debug_build_dir)/$(debug_name) : $(debug_build_dir) $(debug_object_files)
	$(CXX) $(debug_object_files) $(LDFLAGS) -lpthread -o $(debug_build_dir)/$(debug_name)

$(debug_build_dir) :
	mkdir -p $(debug_build_dir)

$(debug_object_dir) :
	mkdir -p $(debug_object_dir)

$(debug_vector_obj): $(src_dir)/vector.ixx | $(debug_object_dir)
	$(CXX) $(DEBUG_CXXFLAGS) -x c++ -c $< -o $@

$(debug_particle_obj): $(src_dir)/particle.ixx $(debug_vector_obj) | $(debug_object_dir)
	$(CXX) $(DEBUG_CXXFLAGS) -x c++ -c $< -o $@

$(debug_system_obj): $(src_dir)/system.ixx $(debug_particle_obj) $(debug_vector_obj) | $(debug_object_dir)
	$(CXX) $(DEBUG_CXXFLAGS) -x c++ -c $< -o $@

$(debug_main_obj): $(src_dir)/main.cpp $(debug_system_obj) $(debug_particle_obj) $(debug_vector_obj) | $(debug_object_dir)
	$(CXX) $(DEBUG_CXXFLAGS) -c $< -o $@

debug_run : $(debug_build_dir)/$(debug_name)
	cd $(debug_build_dir) &&\
	./$(debug_name) $(args)
	cd src &&\
	python3 plot_planets.py

clean :
	rm -rf $(build_dir)
	rm -rf gcm.cache