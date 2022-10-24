#pragma once
#ifndef UTILS_HPP
#define UTILS_HPP
#include "arm_includes.hpp"
#include "bbox.hpp"
#include <float.h>
#include <time.h>

struct ABC{
    static tuple<list<string>, int> get_classes(string classes_path);
    static numcpp::ndarray letterbox(numcpp::ndarray image, tuple<int ,int> expected_size);
    static numcpp::ndarray draw_visual(numcpp::ndarray image, mvector<2, double> __boxes, numcpp::ndarray __sources,
                        numcpp::ndarray __classes, vector<string> class_labels, vector<string> class_colors);
    static numcpp::ndarray preprocessInput(numcpp::ndarray image);
};

numcpp::ndarray draw_line(numcpp::ndarray image, int x, int y, int x1, int y1, 
                    list<double> color, int l = 35, int t = 1);

std::function<int(float)> display_process_time(function<int(float)>);

class TRTModule : public ABC{
protected:
    list<string> classLabels;
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