cmake_minimum_required(VERSION 3.10)
project(GeneralLibs VERSION 0.1)

file(GLOB_RECURSE HEADER_FILES ${PROJECT_SOURCE_DIR}//includes//*.hpp)
file(GLOB_RECURSE SRC_FILES    ${PROJECT_SOURCE_DIR}//src//*.cpp)

add_library(GeneralLibs STATIC ${HEADER_FILES} ${SRC_FILES})

target_include_directories(GeneralLibs PUBLIC 
    $<BUILD_INTERFACE:
        ${PROJECT_SOURCE_DIR}/includes
    >
)
target_compile_features(GeneralLibs PUBLIC cxx_std_20)