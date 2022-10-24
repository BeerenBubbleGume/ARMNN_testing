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
    
    static mvector<2, int> yolo_correct_boxes(mvector<1, int> box_xy, mvector<1, int> box_wh, 
                                tuple<int, int> input_shape, tuple<int, int> image_shape);
    
};
#endif