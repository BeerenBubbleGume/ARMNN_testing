#ifndef UTILS_HPP
#define UTILS_HPP
#include "bbox.hpp"
#include <limits>
struct ABC{
    vector<string> get_classes(string classes_path);
    cv::Mat letterbox(cv::Mat image, vector<float> expected_size);
    nc::NdArray<float> draw_visual(cv::Mat image, nc::NdArray<float> __boxes, nc::NdArray<float> __scores,
                        nc::NdArray<float> __classes, vector<string> class_labels, vector<float> class_colors);
    cv::Mat preprocessInput(cv::Mat image);
};

nc::NdArray<float> draw_line(nc::NdArray<float> image, int x, int y, int x1, int y1, 
                    vector<float> color, int l = 35, int t = 1);

void display_process_time();

enum TRT_INTERFERENCE{
    boxes,
    scores,
    classes
};

class TRTModule : public ABC{
private:
    vector<string> classLabels;
    vector<int64_t> inputNodeDims;
    vector<vector<int64_t>> outputNodeDims;
    vector<float> inputValuesHandler;
    vector<const char*> inputNodeNames;
    vector<const char*> outputNodeNames;
    Ort::AllocatorWithDefaultOptions allocator;
    Ort::SessionOptions session_options;
    Ort::Env* env;
    Ort::MemoryInfo cpuInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeCPU);
    Ort::Session* session;
    /*armnnOnnxParser::IOnnxParserPtr* parser;
    armnn::INetworkPtr* network;
    vector<armnnUtils::TContainer> inputDataContainers;
    armnn::DataLayout inputTensorDataLayout;
    vector<armnn::BindingPointInfo> inputBindings;
    vector<armnn::BindingPointInfo> outputBindings;
    armnn::IRuntime::CreationOptions options;
    armnn::IRuntimePtr* runtime;
    armnn::IOptimizedNetworkPtr* optNet;
    armnn::NetworkId* networkIdentifier;
    vector<armnnUtils::TContainer> outputDataContainers;
    int inputTensorBatchSize;*/
    vector<float> classColors;
    int numNames;
    unsigned inputTensorSize = 1;
    int numOutputs = 1;
    char* inputName;
    vector<float> imageShape;
    bboxes* box;

    vector<nc::NdArray<float>> trtInference(cv::Mat intpuData, list<float> imgz);
    void loadModelAndPredict(string pathModel);
    void initHandlers();
    virtual ~TRTModule();
    
public:
    TRTModule(string pathModel, string pathClasses);
    nc::NdArray<float> extractImage(cv::Mat img);
    void startNN(string videoSrc, string outputPath, int fps);
};

/*armnnOnnxParser::IOnnxParserPtr createParser(){
    return armnnOnnxParser::IOnnxParser::Create();
}

armnn::INetwork createNetworkPtr(string pathModel, armnnOnnxParser::IOnnxParser& parser){
    return parser.CreateNetworkFromBinaryFile(pathModel.c_str());
}

armnn::IRuntime createRuntime(armnn::IRuntime::CreationOptions options){
    return armnn::IRuntime::Create(options);
}

armnn::IOptimizedNetwork optimize(const armnn::INetwork& network, const armnn::IRuntime& runtime){
    return armnn::Optimize(network, {armnn::Compute::CpuAcc, armnn::Compute::CpuRef}, runtime.GetDeviceSpec());
}*/

#endif
