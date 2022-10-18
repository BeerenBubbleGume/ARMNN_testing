#pragma once
#ifndef BBOX_HPP
#define BBOX_HPP
#include "arm_includes.hpp"


class bboxes{
public:
    bboxes();
    virtual ~bboxes();
    
    static numcpp::ndarray yolo_correct_boxes(numcpp::ndarray box_xy, numcpp::ndarray box_wh, 
                                tuple<int, int> input_shape, tuple<int, int> image_shape);
    
};
#endif