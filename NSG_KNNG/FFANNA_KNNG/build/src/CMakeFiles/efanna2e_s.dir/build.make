# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /data/PT4HV/NSG_KNNG/FFANNA_KNNG

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /data/PT4HV/NSG_KNNG/FFANNA_KNNG/build

# Include any dependencies generated for this target.
include src/CMakeFiles/efanna2e_s.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include src/CMakeFiles/efanna2e_s.dir/compiler_depend.make

# Include the progress variables for this target.
include src/CMakeFiles/efanna2e_s.dir/progress.make

# Include the compile flags for this target's objects.
include src/CMakeFiles/efanna2e_s.dir/flags.make

src/CMakeFiles/efanna2e_s.dir/index.cpp.o: src/CMakeFiles/efanna2e_s.dir/flags.make
src/CMakeFiles/efanna2e_s.dir/index.cpp.o: ../src/index.cpp
src/CMakeFiles/efanna2e_s.dir/index.cpp.o: src/CMakeFiles/efanna2e_s.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/data/PT4HV/NSG_KNNG/FFANNA_KNNG/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object src/CMakeFiles/efanna2e_s.dir/index.cpp.o"
	cd /data/PT4HV/NSG_KNNG/FFANNA_KNNG/build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT src/CMakeFiles/efanna2e_s.dir/index.cpp.o -MF CMakeFiles/efanna2e_s.dir/index.cpp.o.d -o CMakeFiles/efanna2e_s.dir/index.cpp.o -c /data/PT4HV/NSG_KNNG/FFANNA_KNNG/src/index.cpp

src/CMakeFiles/efanna2e_s.dir/index.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/efanna2e_s.dir/index.cpp.i"
	cd /data/PT4HV/NSG_KNNG/FFANNA_KNNG/build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /data/PT4HV/NSG_KNNG/FFANNA_KNNG/src/index.cpp > CMakeFiles/efanna2e_s.dir/index.cpp.i

src/CMakeFiles/efanna2e_s.dir/index.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/efanna2e_s.dir/index.cpp.s"
	cd /data/PT4HV/NSG_KNNG/FFANNA_KNNG/build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /data/PT4HV/NSG_KNNG/FFANNA_KNNG/src/index.cpp -o CMakeFiles/efanna2e_s.dir/index.cpp.s

src/CMakeFiles/efanna2e_s.dir/index_graph.cpp.o: src/CMakeFiles/efanna2e_s.dir/flags.make
src/CMakeFiles/efanna2e_s.dir/index_graph.cpp.o: ../src/index_graph.cpp
src/CMakeFiles/efanna2e_s.dir/index_graph.cpp.o: src/CMakeFiles/efanna2e_s.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/data/PT4HV/NSG_KNNG/FFANNA_KNNG/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object src/CMakeFiles/efanna2e_s.dir/index_graph.cpp.o"
	cd /data/PT4HV/NSG_KNNG/FFANNA_KNNG/build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT src/CMakeFiles/efanna2e_s.dir/index_graph.cpp.o -MF CMakeFiles/efanna2e_s.dir/index_graph.cpp.o.d -o CMakeFiles/efanna2e_s.dir/index_graph.cpp.o -c /data/PT4HV/NSG_KNNG/FFANNA_KNNG/src/index_graph.cpp

src/CMakeFiles/efanna2e_s.dir/index_graph.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/efanna2e_s.dir/index_graph.cpp.i"
	cd /data/PT4HV/NSG_KNNG/FFANNA_KNNG/build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /data/PT4HV/NSG_KNNG/FFANNA_KNNG/src/index_graph.cpp > CMakeFiles/efanna2e_s.dir/index_graph.cpp.i

src/CMakeFiles/efanna2e_s.dir/index_graph.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/efanna2e_s.dir/index_graph.cpp.s"
	cd /data/PT4HV/NSG_KNNG/FFANNA_KNNG/build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /data/PT4HV/NSG_KNNG/FFANNA_KNNG/src/index_graph.cpp -o CMakeFiles/efanna2e_s.dir/index_graph.cpp.s

src/CMakeFiles/efanna2e_s.dir/index_kdtree.cpp.o: src/CMakeFiles/efanna2e_s.dir/flags.make
src/CMakeFiles/efanna2e_s.dir/index_kdtree.cpp.o: ../src/index_kdtree.cpp
src/CMakeFiles/efanna2e_s.dir/index_kdtree.cpp.o: src/CMakeFiles/efanna2e_s.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/data/PT4HV/NSG_KNNG/FFANNA_KNNG/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object src/CMakeFiles/efanna2e_s.dir/index_kdtree.cpp.o"
	cd /data/PT4HV/NSG_KNNG/FFANNA_KNNG/build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT src/CMakeFiles/efanna2e_s.dir/index_kdtree.cpp.o -MF CMakeFiles/efanna2e_s.dir/index_kdtree.cpp.o.d -o CMakeFiles/efanna2e_s.dir/index_kdtree.cpp.o -c /data/PT4HV/NSG_KNNG/FFANNA_KNNG/src/index_kdtree.cpp

src/CMakeFiles/efanna2e_s.dir/index_kdtree.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/efanna2e_s.dir/index_kdtree.cpp.i"
	cd /data/PT4HV/NSG_KNNG/FFANNA_KNNG/build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /data/PT4HV/NSG_KNNG/FFANNA_KNNG/src/index_kdtree.cpp > CMakeFiles/efanna2e_s.dir/index_kdtree.cpp.i

src/CMakeFiles/efanna2e_s.dir/index_kdtree.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/efanna2e_s.dir/index_kdtree.cpp.s"
	cd /data/PT4HV/NSG_KNNG/FFANNA_KNNG/build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /data/PT4HV/NSG_KNNG/FFANNA_KNNG/src/index_kdtree.cpp -o CMakeFiles/efanna2e_s.dir/index_kdtree.cpp.s

src/CMakeFiles/efanna2e_s.dir/index_random.cpp.o: src/CMakeFiles/efanna2e_s.dir/flags.make
src/CMakeFiles/efanna2e_s.dir/index_random.cpp.o: ../src/index_random.cpp
src/CMakeFiles/efanna2e_s.dir/index_random.cpp.o: src/CMakeFiles/efanna2e_s.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/data/PT4HV/NSG_KNNG/FFANNA_KNNG/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object src/CMakeFiles/efanna2e_s.dir/index_random.cpp.o"
	cd /data/PT4HV/NSG_KNNG/FFANNA_KNNG/build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT src/CMakeFiles/efanna2e_s.dir/index_random.cpp.o -MF CMakeFiles/efanna2e_s.dir/index_random.cpp.o.d -o CMakeFiles/efanna2e_s.dir/index_random.cpp.o -c /data/PT4HV/NSG_KNNG/FFANNA_KNNG/src/index_random.cpp

src/CMakeFiles/efanna2e_s.dir/index_random.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/efanna2e_s.dir/index_random.cpp.i"
	cd /data/PT4HV/NSG_KNNG/FFANNA_KNNG/build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /data/PT4HV/NSG_KNNG/FFANNA_KNNG/src/index_random.cpp > CMakeFiles/efanna2e_s.dir/index_random.cpp.i

src/CMakeFiles/efanna2e_s.dir/index_random.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/efanna2e_s.dir/index_random.cpp.s"
	cd /data/PT4HV/NSG_KNNG/FFANNA_KNNG/build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /data/PT4HV/NSG_KNNG/FFANNA_KNNG/src/index_random.cpp -o CMakeFiles/efanna2e_s.dir/index_random.cpp.s

# Object files for target efanna2e_s
efanna2e_s_OBJECTS = \
"CMakeFiles/efanna2e_s.dir/index.cpp.o" \
"CMakeFiles/efanna2e_s.dir/index_graph.cpp.o" \
"CMakeFiles/efanna2e_s.dir/index_kdtree.cpp.o" \
"CMakeFiles/efanna2e_s.dir/index_random.cpp.o"

# External object files for target efanna2e_s
efanna2e_s_EXTERNAL_OBJECTS =

src/libefanna2e_s.a: src/CMakeFiles/efanna2e_s.dir/index.cpp.o
src/libefanna2e_s.a: src/CMakeFiles/efanna2e_s.dir/index_graph.cpp.o
src/libefanna2e_s.a: src/CMakeFiles/efanna2e_s.dir/index_kdtree.cpp.o
src/libefanna2e_s.a: src/CMakeFiles/efanna2e_s.dir/index_random.cpp.o
src/libefanna2e_s.a: src/CMakeFiles/efanna2e_s.dir/build.make
src/libefanna2e_s.a: src/CMakeFiles/efanna2e_s.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/data/PT4HV/NSG_KNNG/FFANNA_KNNG/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Linking CXX static library libefanna2e_s.a"
	cd /data/PT4HV/NSG_KNNG/FFANNA_KNNG/build/src && $(CMAKE_COMMAND) -P CMakeFiles/efanna2e_s.dir/cmake_clean_target.cmake
	cd /data/PT4HV/NSG_KNNG/FFANNA_KNNG/build/src && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/efanna2e_s.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
src/CMakeFiles/efanna2e_s.dir/build: src/libefanna2e_s.a
.PHONY : src/CMakeFiles/efanna2e_s.dir/build

src/CMakeFiles/efanna2e_s.dir/clean:
	cd /data/PT4HV/NSG_KNNG/FFANNA_KNNG/build/src && $(CMAKE_COMMAND) -P CMakeFiles/efanna2e_s.dir/cmake_clean.cmake
.PHONY : src/CMakeFiles/efanna2e_s.dir/clean

src/CMakeFiles/efanna2e_s.dir/depend:
	cd /data/PT4HV/NSG_KNNG/FFANNA_KNNG/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /data/PT4HV/NSG_KNNG/FFANNA_KNNG /data/PT4HV/NSG_KNNG/FFANNA_KNNG/src /data/PT4HV/NSG_KNNG/FFANNA_KNNG/build /data/PT4HV/NSG_KNNG/FFANNA_KNNG/build/src /data/PT4HV/NSG_KNNG/FFANNA_KNNG/build/src/CMakeFiles/efanna2e_s.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : src/CMakeFiles/efanna2e_s.dir/depend

