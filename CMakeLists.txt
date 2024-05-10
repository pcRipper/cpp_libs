cmake_minimum_required(VERSION 3.10)
project(LeetCodeLibs VERSION 0.1)


file(GLOB_RECURSE HEADER_FILES ".//Headers//*.hpp")
file(GLOB_RECURSE SOURCE_FILES ".//Source//*.cpp")

add_library(LeetCodeLibs STATIC ${HEADER_FILES} ${SOURCE_FILES})

target_include_directories(LeetCodeLibs PUBLIC ".//Headers")
target_compile_features(LeetCodeLibs PUBLIC cxx_std_17)