#ifndef BBOX_HPP
#define BBOX_HPP
#include "arm_includes.hpp"

class bboxes{
public:
    bboxes();
    virtual ~bboxes();
    
    static nc::NdArray<float>  yolo_correct_boxes(nc::NdArray<float>  box_xy, nc::NdArray<float>  box_wh, 
                                vector<float> input_shape, list<float> image_shape);
    vector<nc::NdArray<float>> preprocess(nc::NdArray<Ort::Value> output, vector<float> input_shape, list<float> image_shape);

};
#endif