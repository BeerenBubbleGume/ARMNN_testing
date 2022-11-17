#ifndef UTILS_HPP
#define UTILS_HPP
#include "bbox.hpp"
struct ABC{
    static vector<string> get_classes(string classes_path);
    static nc::NdArray<float> letterbox(nc::NdArray<float> image, tuple<int ,int> expected_size);
    static nc::NdArray<float> draw_visual(nc::NdArray<float> image, nc::NdArray<float> __boxes, nc::NdArray<float> __scores,
                        nc::NdArray<float> __classes, vector<string> class_labels, vector<float> class_colors);
    static nc::NdArray<float> preprocessInput(nc::NdArray<float> image);
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
protected:
    vector<string> classLabels;
    string inputName;
    string outputName;
    string inVideoName;
    Ort::Experimental::Session* session;
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
    vector<float> imageShape;
    bboxes* box;

    vector<nc::NdArray<float>> trtInference(nc::NdArray<float> intpuData, nc::NdArray<float> imgz);
    void loadModelAndPredict(string pathModel);
    virtual ~TRTModule();
    
public:
    TRTModule(string pathModel, string pathClasses);
    nc::NdArray<float> extractImage(nc::NdArray<float> img);
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
