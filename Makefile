CXX ?= g++

BUILD ?= debug
BUILD_DIR := build/$(BUILD)
OBJECT_DIR := $(BUILD_DIR)/obj
BINARY_DIR := $(BUILD_DIR)/bin

EXAMPLE_NAME := $(BINARY_DIR)/LanguageModels.Examples
TEST_NAME := $(BINARY_DIR)/LanguageModels.Tests

# LanguageModels itself is header-only (every implementation file is a
# class template), so there's no library source to compile — only the
# include paths below are needed to build the examples and the tests.
INCLUDE_DIRS := \
	-ILanguageModels/include/common \
	-ILanguageModels/include/config \
	-ILanguageModels/include/layers \
	-ILanguageModels/include/embeddings \
	-ILanguageModels/include/metrics \
	-ILanguageModels/include/normalizations \
	-ILanguageModels/include/models \
	-ILanguageModels/include/tokenizers \
	-ILanguageModels.Tests

CPPFLAGS := $(INCLUDE_DIRS) -MMD -MP
COMMON_FLAGS := -std=c++20 -Wall -Wextra -Wpedantic

ifeq ($(BUILD),release)
	CXXFLAGS := $(COMMON_FLAGS) -O3 -DNDEBUG
else
	CXXFLAGS := $(COMMON_FLAGS) -O0 -g
endif

# test_simple_rnn.cpp is intentionally excluded here too, matching the
# Visual Studio project: it's a standalone earlier draft, not wired into
# the build, and defines its own main() that would collide with Main.cpp.
EXAMPLE_SOURCES := $(filter-out LanguageModels.Examples/test_simple_rnn.cpp,$(wildcard LanguageModels.Examples/*.cpp))
EXAMPLE_OBJECTS := $(patsubst %.cpp,$(OBJECT_DIR)/%.o,$(EXAMPLE_SOURCES))
RESOURCE_FILES := $(wildcard resources/*.txt)

TEST_SOURCES := $(shell find LanguageModels.Tests -type f -name '*.cpp')
TEST_OBJECTS := $(patsubst %.cpp,$(OBJECT_DIR)/%.o,$(TEST_SOURCES))

# Google Test (e.g. `apt install libgtest-dev`).
GTEST_LIBS ?= -lgtest_main -lgtest -pthread

DEPENDENCY_FILES := $(EXAMPLE_OBJECTS:.o=.d) $(TEST_OBJECTS:.o=.d)

.PHONY: all library examples tests test run clean help

all: examples

# Kept only so `make library` doesn't error for anyone using the same
# command vocabulary as the sibling MachineLearningModels repo.
library:
	@echo "LanguageModels is header-only; nothing to build."

# TestMiniTransformer.cpp/TestWordPieceTokenizer.cpp locate their resource
# files by bare filename, so they need to sit next to the binary.
examples: $(EXAMPLE_NAME)
	@cp -f $(RESOURCE_FILES) $(BINARY_DIR)/

tests: $(TEST_NAME)

test: $(TEST_NAME)
	$(TEST_NAME)

run: examples
	cd $(BINARY_DIR) && ./LanguageModels.Examples $(ARGS)

$(EXAMPLE_NAME): $(EXAMPLE_OBJECTS)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $^ -o $@

$(TEST_NAME): $(TEST_OBJECTS)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $^ $(GTEST_LIBS) -o $@

$(OBJECT_DIR)/%.o: %.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

clean:
	rm -rf build

help:
	@echo "make                 Build the example executable (debug)"
	@echo "make BUILD=release   Build an optimized release binary"
	@echo "make examples        Build the example executable"
	@echo "make run             Build and run it (full Mini Transformer dataset)"
	@echo "make run ARGS=--quick   Same, with the small 100-pair dataset"
	@echo "make tests           Build the Google Test executable"
	@echo "make test            Build and run all tests"
	@echo "make clean           Remove Makefile build outputs"

-include $(DEPENDENCY_FILES)
