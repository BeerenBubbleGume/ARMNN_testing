cmake_minimum_required(VERSION 3.13)

project(ARMNN_test VERSION 0.0 LANGUAGES CXX)

set(CMAKE_C_COMPILER "aarch64-linux-gnu-gcc-11")
set(CMAKE_CXX_COMPILER "aarch64-linux-gnu-g++-11")

add_compile_options(
    #-Werror
    -Wall
    -Wextra
    -std=gnu++20
    -Wpedantic
    -g
    #-Wcast-align
    #-Wcast-qual
    #-Wconversion
    #-Wenum-compare
    #-Wfloat-equal
    #-Wnon-virtual-dtor
    #-Wold-style-cast
    #-Woverloaded-virtual
)

add_executable(ARMNN_test   "main.cpp"
                            "src/source/utils.cpp"
                            "src/source/bbox.cpp"
                            #"src/include/ImageTensorGenerator.hpp"
                            "src/source/VerificationHelpers.cpp"
                            #"src/include/arm_includes.hpp"
                            "src/source/InferenceImage.cpp"
                            )

set_property(TARGET ARMNN_test PROPERTY CXX_STANDARD 20)

if(NOT CMAKE_CXX_EXTENSIONS)
    set(CMAKE_CXX_EXTENSIONS OFF)
endif()

add_subdirectory("src")

#target_include_directories(ARMNN_test /usr/include)
#find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui video imgcodecs videoio features2d dnn gapi calib3d objdetect ml)
#find_package(OpenCV REQUIRED)
include_directories("/usr/local/include/")
include_directories("/usr/local/include/opencv4")
target_link_libraries(ARMNN_test
                        opencv_calib3d
                        opencv_core
                        opencv_dnn
                        opencv_features2d
                        opencv_flann
                        opencv_gapi
                        opencv_highgui
                        opencv_imgcodecs
                        opencv_imgproc
                        opencv_ml
                        opencv_objdetect
                        opencv_photo
                        opencv_stitching
                        opencv_videoio
                        opencv_video
                        protobuf
                        fmt
                        armnn
                        armnnOnnxParser
                        armnnUtils
                        armnnBasePipeServer
                        armnnTestUtils
                        timelineDecoderJson
                        timelineDecoder
                        armnnBasePipeServer)
