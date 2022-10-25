#pragma once
#ifndef UTILS_HPP
#define UTILS_HPP
#include "arm_includes.hpp"
#include "bbox.hpp"
#include <float.h>
#include <time.h>

struct ABC{
    static vector<string> get_classes(string classes_path);
    static nc::NdArray<int> letterbox(nc::NdArray<int> image, tuple<int ,int> expected_size);
    static nc::NdArray<double> draw_visual(nc::NdArray<double> image, nc::NdArray<double> __boxes, nc::NdArray<double> __sources,
                        nc::NdArray<string> __classes, vector<string> class_labels, vector<double> class_colors);
    static nc::NdArray<double> preprocessInput(nc::NdArray<double> image);
};

nc::NdArray<double> draw_line(nc::NdArray<double> image, int x, int y, int x1, int y1, 
                    list<double> color, int l = 35, int t = 1);

void display_process_time();

void setVaraibles(vector<void> inData, vector<void> outArray);

class TRTModule : public ABC{
protected:
    vector<string> classLabels;
    const string inputName;
    const string outputName;
    int numNames;
    vector<void> imageShape;
    bboxes* bboxes;

    map<string, vector<vector<void>>> trtInference(vector<vector<void>> intpuData, vector<vector<void>> imgz);
    void startNN(string videoSrc, string outputPath, int fps);
    void loadModelAndPredict(string pathModel);
    virtual ~TRTModule();

public:
    TRTModule(string pathModel, string pathClasses);
    vector<vector<void>> extractImage(auto img);

};

#endif