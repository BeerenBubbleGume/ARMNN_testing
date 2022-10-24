#pragma once
#ifndef BBOX_HPP
#define BBOX_HPP
#include "arm_includes.hpp"

template<size_t dimcount, typename T>
struct mvector
{
    typedef std::vector< typename mvector<dimcount-1, T>::type > type;
};

template<typename T>
struct mvector<0,T>
{
    typedef T type;
};

class bboxes{
public:
    bboxes();
    virtual ~bboxes();
    
    static nc::NdArray<int>  yolo_correct_boxes(nc::NdArray<int>  box_xy, nc::NdArray<int>  box_wh, 
                                tuple<int, int> input_shape, tuple<int, int> image_shape);
    
};
#endif