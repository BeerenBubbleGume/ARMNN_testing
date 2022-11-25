#include "src/include/utils.hpp"

int main(int argc, char* argv[]){
    if(argc > 1){
        Ort::SessionOptions session_options;
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "example-model-explorer");
        TRTModule* nn = new TRTModule("weights/yolox-tiny640.onnx", "classes.txt", session_options, env);
        nn->startNN(argv[1], argv[2], 35);
        display_process_time();
        return 0;
    }
    else {
        Ort::SessionOptions session_options;
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "example-model-explorer");
        TRTModule* nn = new TRTModule("weights/yolox-tiny640.onnx", "classes.txt", session_options, env);
        nn->startNN("video/v.avi", "v-out.avi", 35);
        display_process_time();
    }
    return -1;
}