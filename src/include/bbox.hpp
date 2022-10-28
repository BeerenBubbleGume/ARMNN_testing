#pragma once
#ifndef BBOX_HPP
#define BBOX_HPP
#include "arm_includes.hpp"

class bboxes{
public:
    bboxes();
    virtual ~bboxes();
    
    static nc::NdArray<float>  yolo_correct_boxes(nc::NdArray<float>  box_xy, nc::NdArray<float>  box_wh, 
                                vector<float> input_shape, nc::NdArray<float> image_shape);
    list<nc::NdArray<float>> preprocess(nc::NdArray<int> output, tuple<int, int> image_data, tuple<int, int> image_shape);

};
#endif