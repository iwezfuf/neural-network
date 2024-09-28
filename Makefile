# Makefile to compile all .cpp files in the current directory and run main.cpp

# Compiler
CXX = g++

# Flags for the compiler
CXXFLAGS = -Wall -Wextra -g -std=c++20

# Get all .cpp files in the current directory
SRC = $(wildcard src/*.cpp)

# Output executable
TARGET = main

# Compile all .cpp files and create the executable
all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SRC)

# Run the executable
run: $(TARGET)
	./$(TARGET)

# Clean the directory by removing compiled files
clean:
	rm -f $(TARGET)
