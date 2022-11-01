#pragma once
#include "src/include/utils.hpp"

int main(int argc, char* argv[]){
    if(argc > 0){
        TRTModule* nn = new TRTModule("weights/yolox-tiny640.onnx", "classes.txt");
        nn->startNN(argv[0], argv[1], 35);
        display_process_time();
        return 0;
    }
    return -1;
}