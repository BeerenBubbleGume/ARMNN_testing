#pragma once
#ifndef UTILS_HPP
#define UTILS_HPP
#include "arm_includes.hpp"
#include "bbox.hpp"
#include <float.h>
#include <time.h>
#include "../include/ImageTensorGenerator.hpp"
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

void setVaraibles(vector<void> inData, vector<void> outArray);

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
    vector<armnnUtils::TContainer> inputDataContainers;
    armnn::DataLayout inputTensorDataLayout;
    vector<armnn::BindingPointInfo> inputBindings;
    vector<armnn::BindingPointInfo> outputBindings;
    int inputTensorBatchSize;
    vector<float> classColors;
    int numNames;
    vector<unsigned int> imageShape;
    bboxes* box;

    nc::NdArray<float> trtInference(nc::NdArray<float> intpuData, nc::NdArray<float> imgz, int TRT_INTERFERENCE);
    void loadModelAndPredict(string pathModel);
    virtual ~TRTModule();

public:
    TRTModule(string pathModel, string pathClasses);
    nc::NdArray<float> extractImage(nc::NdArray<float> img);
    void startNN(string videoSrc, string outputPath, int fps);
};

#endif