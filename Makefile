# Makefile for nbody simulation
args ?= -f ../src/solar_system.txt

name = nbody

src_dir = src
src_files = $(wildcard $(src_dir)/*.cpp)

build_dir = build
object_dir = $(build_dir)/obj
object_files = $(patsubst $(src_dir)/%.cpp,$(object_dir)/%.o,$(src_files))

.PHONY: all run clean debug debug_run

all : $(build_dir)/$(name)

$(build_dir)/$(name) : $(build_dir) $(object_files)
	g++ $(object_files) -fopenmp -o $(build_dir)/$(name)

$(build_dir) :
	mkdir $(build_dir)

$(object_dir) :
	mkdir $(object_dir)

$(object_dir)/%.o: $(src_dir)/%.cpp | $(object_dir)
	g++ -fopenmp -c $< -o $@

run : $(build_dir)/$(name)
	cd $(build_dir) &&\
	./$(name) $(args)
	cd src &&\
	python3 plt_trajectory.py 

# Debug rules
debug_name = $(name)_debug
debug_build_dir = $(build_dir)/debug
debug_object_dir = $(debug_build_dir)/obj
debug_object_files = $(patsubst $(src_dir)/%.cpp,$(debug_object_dir)/%.o,$(src_files))

debug : $(debug_build_dir)/$(debug_name)

$(debug_build_dir)/$(debug_name) : $(debug_build_dir) $(debug_object_files)
	g++ $(debug_object_files) -o $(debug_build_dir)/$(debug_name) -g -fopenmp -lpthread

$(debug_build_dir) :
	mkdir -p $(debug_build_dir)

$(debug_object_dir) :
	mkdir -p $(debug_object_dir)

$(debug_object_dir)/%.o: $(src_dir)/%.cpp | $(debug_object_dir)
	g++ -c $< -o $@ -g

debug_run : $(debug_build_dir)/$(debug_name)
	cd $(debug_build_dir) &&\
	./$(debug_name) $(args)
	cd src &&\
	python3 plot_planets.py

clean :
	rm -rf $(build_dir)