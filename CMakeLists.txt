cmake_minimum_required(VERSION 3.10)
project(GeneralLibs VERSION 0.1)


file(GLOB_RECURSE HEADER_FILES ".//Headers//*.hpp")
file(GLOB_RECURSE SOURCE_FILES ".//Source//*.cpp")

add_library(GeneralLibs STATIC ${HEADER_FILES} ${SOURCE_FILES})

target_include_directories(GeneralLibs PUBLIC ".//Headers")
target_compile_features(GeneralLibs PUBLIC cxx_std_20)