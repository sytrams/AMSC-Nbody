# Makefile for nbody simulation
args ?= -f ../src/solar_system.txt
test_args ?= -f src/solar_system.txt

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

debug_name = $(name)_debug
debug_build_dir = $(build_dir)/debug
debug_object_dir = $(debug_build_dir)/obj
debug_main_obj = $(debug_object_dir)/main.o
debug_vector_obj = $(debug_object_dir)/vector.o
debug_particle_obj = $(debug_object_dir)/particle.o
debug_system_obj = $(debug_object_dir)/system.o
debug_tree_obj = $(debug_object_dir)/tree.o
debug_object_files = $(debug_tree_obj) $(debug_vector_obj) $(debug_particle_obj) $(debug_system_obj) $(debug_main_obj)
debug_module_dir = $(debug_build_dir)/pcm
debug_vector_pcm = $(debug_module_dir)/vector.pcm
debug_particle_pcm = $(debug_module_dir)/particle.pcm
debug_system_pcm = $(debug_module_dir)/system.pcm
debug_tree_pcm = $(debug_module_dir)/tree.pcm

UNAME_S := $(shell uname -s)

ifeq ($(UNAME_S),Darwin)
	CXX := /opt/homebrew/opt/llvm@18/bin/clang++
	SDKROOT ?= $(shell xcrun --show-sdk-path)
	OMP_PREFIX ?= /opt/homebrew/opt/libomp
	CXXFLAGS = -std=c++23 -stdlib=libc++ -fopenmp -I$(OMP_PREFIX)/include -isysroot $(SDKROOT)
	LDFLAGS = -stdlib=libc++ -L$(OMP_PREFIX)/lib -Wl,-rpath,$(OMP_PREFIX)/lib -fopenmp
	vector_import = -fmodule-file=vector=$(vector_pcm)
	particle_imports = $(vector_import)
	system_imports = $(vector_import) -fmodule-file=particle=$(particle_pcm)
	tree_imports = $(system_imports) -fmodule-file=system=$(system_pcm)
	main_imports = $(tree_imports)
	debug_vector_import = -fmodule-file=vector=$(debug_vector_pcm)
	debug_particle_imports = $(debug_vector_import)
	debug_system_imports = $(debug_vector_import) -fmodule-file=particle=$(debug_particle_pcm)
	debug_tree_imports = $(debug_system_imports) -fmodule-file=system=$(debug_system_pcm)
	debug_main_imports = $(debug_tree_imports)
else
	CXX ?= g++
	CXXFLAGS = -std=c++23 -fmodules-ts -fopenmp
	LDFLAGS = -fopenmp
endif

DEBUG_CXXFLAGS = $(CXXFLAGS) -g

.PHONY: all run test-run plot-images clean debug debug_run compile-commands intellisense

compile-commands: compile_commands.json

intellisense: compile_commands.json $(system_obj)

compile_commands.json: Makefile
	@printf '[\n' > $@
ifeq ($(UNAME_S),Darwin)
	@printf '  {\n    "directory": "%s",\n    "file": "%s/src/vector.cppm",\n    "command": "%s %s -c %s/src/vector.cppm -o %s/%s -fmodule-output=%s/%s"\n  },\n' \
		"$(CURDIR)" "$(CURDIR)" "$(CXX)" "$(CXXFLAGS)" "$(CURDIR)" "$(CURDIR)" "$(vector_obj)" "$(CURDIR)" "$(vector_pcm)" >> $@
	@printf '  {\n    "directory": "%s",\n    "file": "%s/src/particle.cppm",\n    "command": "%s %s -c %s/src/particle.cppm -o %s/%s -fmodule-output=%s/%s %s"\n  },\n' \
		"$(CURDIR)" "$(CURDIR)" "$(CXX)" "$(CXXFLAGS)" "$(CURDIR)" "$(CURDIR)" "$(particle_obj)" "$(CURDIR)" "$(particle_pcm)" "$(particle_imports)" >> $@
	@printf '  {\n    "directory": "%s",\n    "file": "%s/src/system.cppm",\n    "command": "%s %s -c %s/src/system.cppm -o %s/%s -fmodule-output=%s/%s %s"\n  },\n' \
		"$(CURDIR)" "$(CURDIR)" "$(CXX)" "$(CXXFLAGS)" "$(CURDIR)" "$(CURDIR)" "$(system_obj)" "$(CURDIR)" "$(system_pcm)" "$(system_imports)" >> $@
	@printf '  {\n    "directory": "%s",\n    "file": "%s/src/tree.cppm",\n    "command": "%s %s -c %s/src/tree.cppm -o %s/%s -fmodule-output=%s/%s %s"\n  },\n' \
		"$(CURDIR)" "$(CURDIR)" "$(CXX)" "$(CXXFLAGS)" "$(CURDIR)" "$(CURDIR)" "$(tree_obj)" "$(CURDIR)" "$(tree_pcm)" "$(tree_imports)" >> $@
	@printf '  {\n    "directory": "%s",\n    "file": "%s/src/main.cpp",\n    "command": "%s %s -c %s/src/main.cpp -o %s/%s %s"\n  }\n' \
		"$(CURDIR)" "$(CURDIR)" "$(CXX)" "$(CXXFLAGS)" "$(CURDIR)" "$(CURDIR)" "$(main_obj)" "$(main_imports)" >> $@
else
	@printf '  {\n    "directory": "%s",\n    "file": "%s/src/vector.cppm",\n    "command": "%s %s -x c++ -c %s/src/vector.cppm -o %s/%s"\n  },\n' \
		"$(CURDIR)" "$(CURDIR)" "$(CXX)" "$(CXXFLAGS)" "$(CURDIR)" "$(CURDIR)" "$(vector_obj)" >> $@
	@printf '  {\n    "directory": "%s",\n    "file": "%s/src/particle.cppm",\n    "command": "%s %s -x c++ -c %s/src/particle.cppm -o %s/%s"\n  },\n' \
		"$(CURDIR)" "$(CURDIR)" "$(CXX)" "$(CXXFLAGS)" "$(CURDIR)" "$(CURDIR)" "$(particle_obj)" >> $@
	@printf '  {\n    "directory": "%s",\n    "file": "%s/src/system.cppm",\n    "command": "%s %s -x c++ -c %s/src/system.cppm -o %s/%s"\n  },\n' \
		"$(CURDIR)" "$(CURDIR)" "$(CXX)" "$(CXXFLAGS)" "$(CURDIR)" "$(CURDIR)" "$(system_obj)" >> $@
	@printf '  {\n    "directory": "%s",\n    "file": "%s/src/tree.cppm",\n    "command": "%s %s -x c++ -c %s/src/tree.cppm -o %s/%s"\n  },\n' \
		"$(CURDIR)" "$(CURDIR)" "$(CXX)" "$(CXXFLAGS)" "$(CURDIR)" "$(CURDIR)" "$(tree_obj)" >> $@
	@printf '  {\n    "directory": "%s",\n    "file": "%s/src/main.cpp",\n    "command": "%s %s -c %s/src/main.cpp -o %s/%s"\n  }\n' \
		"$(CURDIR)" "$(CURDIR)" "$(CXX)" "$(CXXFLAGS)" "$(CURDIR)" "$(CURDIR)" "$(main_obj)" >> $@
endif
	@printf ']\n' >> $@

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
else
$(vector_obj): $(src_dir)/vector.cppm | $(object_dir)
	$(CXX) $(CXXFLAGS) -x c++ -c $< -o $@

$(particle_obj): $(src_dir)/particle.cppm $(vector_obj) | $(object_dir)
	$(CXX) $(CXXFLAGS) -x c++ -c $< -o $@

$(system_obj): $(src_dir)/system.cppm $(particle_obj) $(vector_obj) | $(object_dir)
	$(CXX) $(CXXFLAGS) -x c++ -c $< -o $@

$(tree_obj): $(src_dir)/tree.cppm $(particle_obj) $(vector_obj) $(system_obj) | $(object_dir)
	$(CXX) $(CXXFLAGS) -x c++ -c $< -o $@

$(main_obj): $(src_dir)/main.cpp $(system_obj) $(particle_obj) $(vector_obj) | $(object_dir)
	$(CXX) $(CXXFLAGS) -c $< -o $@
endif

plot-images :
	python3 $(src_dir)/plt_trajectory.py
	python3 $(src_dir)/plot_planets.py

run : $(build_dir)/$(name)
	cd $(build_dir) && ./$(name) $(args)
	$(MAKE) plot-images

test-run : $(build_dir)/$(name)
	./$(build_dir)/$(name) $(test_args)
	$(MAKE) plot-images

debug : $(debug_build_dir)/$(debug_name)

$(debug_build_dir)/$(debug_name) : $(debug_build_dir) $(debug_object_files)
	$(CXX) $(debug_object_files) $(LDFLAGS) -o $@

$(debug_build_dir) :
	mkdir -p $(debug_build_dir)

$(debug_object_dir) :
	mkdir -p $(debug_object_dir)

ifeq ($(UNAME_S),Darwin)
$(debug_module_dir) :
	mkdir -p $(debug_module_dir)

$(debug_vector_obj): $(src_dir)/vector.cppm | $(debug_object_dir) $(debug_module_dir)
	$(CXX) $(DEBUG_CXXFLAGS) -c $< -o $@ -fmodule-output=$(debug_vector_pcm)

$(debug_particle_obj): $(src_dir)/particle.cppm $(debug_vector_obj) | $(debug_object_dir) $(debug_module_dir)
	$(CXX) $(DEBUG_CXXFLAGS) -c $< -o $@ -fmodule-output=$(debug_particle_pcm) $(debug_particle_imports)

$(debug_system_obj): $(src_dir)/system.cppm $(debug_particle_obj) $(debug_vector_obj) | $(debug_object_dir) $(debug_module_dir)
	$(CXX) $(DEBUG_CXXFLAGS) -c $< -o $@ -fmodule-output=$(debug_system_pcm) $(debug_system_imports)

$(debug_tree_obj): $(src_dir)/tree.cppm $(debug_system_obj) $(debug_particle_obj) $(debug_vector_obj) | $(debug_object_dir) $(debug_module_dir)
	$(CXX) $(DEBUG_CXXFLAGS) -c $< -o $@ -fmodule-output=$(debug_tree_pcm) $(debug_tree_imports)

$(debug_main_obj): $(src_dir)/main.cpp $(debug_system_obj) $(debug_particle_obj) $(debug_vector_obj) | $(debug_object_dir) $(debug_module_dir)
	$(CXX) $(DEBUG_CXXFLAGS) -c $< -o $@ $(debug_main_imports)
else
$(debug_vector_obj): $(src_dir)/vector.cppm | $(debug_object_dir)
	$(CXX) $(DEBUG_CXXFLAGS) -x c++ -c $< -o $@

$(debug_particle_obj): $(src_dir)/particle.cppm $(debug_vector_obj) | $(debug_object_dir)
	$(CXX) $(DEBUG_CXXFLAGS) -x c++ -c $< -o $@

$(debug_system_obj): $(src_dir)/system.cppm $(debug_particle_obj) $(debug_vector_obj) | $(debug_object_dir)
	$(CXX) $(DEBUG_CXXFLAGS) -x c++ -c $< -o $@

$(debug_tree_obj): $(src_dir)/tree.cppm $(debug_system_obj) $(debug_particle_obj) $(debug_vector_obj) | $(debug_object_dir)
	$(CXX) $(DEBUG_CXXFLAGS) -x c++ -c $< -o $@

$(debug_main_obj): $(src_dir)/main.cpp $(debug_system_obj) $(debug_particle_obj) $(debug_vector_obj) | $(debug_object_dir)
	$(CXX) $(DEBUG_CXXFLAGS) -c $< -o $@
endif

debug_run : $(debug_build_dir)/$(debug_name)
	cd $(debug_build_dir) && ./$(debug_name) $(args)
	$(MAKE) plot-images

clean :
	rm -rf $(build_dir)
	rm -rf gcm.cache
	rm -rf .cache/clangmodules
