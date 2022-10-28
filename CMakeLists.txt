cmake_minimum_required(VERSION 3.13)

project(ARMNN_test VERSION 0.0 LANGUAGES CXX)

set(CMAKE_C_COMPILER "gcc-11.2.0")
set(CMAKE_CXX_COMPILER "/usr/bin/g++-11")

add_compile_options(
    -Werror
    -Wall
    -Wextra
    -Wpedantic

    -Wcast-align
    -Wcast-qual
    -Wconversion
    -Wenum-compare
    -Wfloat-equal
    -Wnon-virtual-dtor
    -Wold-style-cast
    -Woverloaded-virtual
)

add_executable(ARMNN_test   "main.cpp"
                            "src/source/bbox.cpp"
                            "src/source/utils.cpp"
                            "src/source/VerificationHelpers.cpp"
                            "src/include/arm_includes.hpp"
                            "src/include/ImageTensorGenerator.hpp"
                            "src/include/InferanceImage.hpp")

set_property(TARGET ARMNN_test PROPERTY CXX_STANDARD 20)

if(NOT CMAKE_CXX_EXTENSIONS)
    set(CMAKE_CXX_EXTENSIONS OFF)
endif()

add_subdirectory("src")

#target_include_directories(ARMNN_test /usr/include)
#find_library(BOOST)
target_link_libraries(ARMNN_test
                        armnn
                        timelineDecoderJson
                        timelineDecoder
                        armnnBasePipeServer
                        opencv4
                        NumCpp
                        boost)
