#include "src/include/utils.hpp"

int main(int argc, char* argv[]){
    if(argc > 1){
        TRTModule* nn = new TRTModule("weights/yolox-tiny.onnx", "classes.txt");
        nn->startNN(argv[1], argv[2], 35);
        display_process_time();
        return 0;
    }
    else {
        TRTModule* nn = new TRTModule("weights/yolox-tiny.onnx", "classes.txt");
        nn->startNN("video/v.avi", "v-out.avi", 35);
        display_process_time();
    }
    return -1;
}