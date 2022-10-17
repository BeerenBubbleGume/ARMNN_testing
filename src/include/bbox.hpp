#pragma once
#ifndef BBOX_HPP
#define BBOX_HPP
#include "arm_includes.hpp"


class bbox{
public:
    bbox();
    virtual ~bbox();

    static yolo_correct_boxes(vector<vector<auto>> box_xy, vector<vector<auto>> box_wh, 
                                tuple<auto, auto, ...> input_shape, tuple<auto, auto, ...> image_shape);
    
};
#endif