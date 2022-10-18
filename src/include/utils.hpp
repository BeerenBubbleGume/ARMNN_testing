#pragma once
#ifndef UTILS_HPP
#define UTILS_HPP
#include "arm_includes.hpp"

struct ABC{
    static tuple<list<string>, int> get_classes(string classes_path);
    static numcpp::ndarray letterbox(numcpp::ndarray image, tuple<int ,int> expected_size);
    static numcpp::ndarray draw_visual(numcpp::ndarray image, numcpp::ndarray __boxes, numcpp::ndarray __sources,
                        numcpp::ndarray __classes, list<string> class_labels, list<string> class_colors);
    static numcpp::ndarray preprocessInput(numcpp::ndarray image);
};

numcpp::ndarray draw_line(numcpp::ndarray image, int x, int y, int x1, int y1, 
                    list<double> color, int l = 35, int t = 1);

std::function<int(float)> display_process_time(function<int(float)>);
#endif