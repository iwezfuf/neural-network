# CMake minimum version
cmake_minimum_required(VERSION 3.10)

# Project name
project(MainProject)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Set compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -g")

# Include all .cpp files in the src directory
file(GLOB SRC_FILES "src/*.cpp")

# Add the executable
add_executable(main ${SRC_FILES})

# Custom target to run the executable
add_custom_target(run
    COMMAND main
    DEPENDS main
    WORKING_DIRECTORY ${CMAKE_PROJECT_DIR}
)
