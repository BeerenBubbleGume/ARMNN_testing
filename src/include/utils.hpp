#pragma once
#ifndef UTILS_HPP
#define UTILS_HPP
#include "arm_includes.hpp"

struct ABC(){
    static tuple<list<string>, int> get_classes(string classes_path);
    static vector<auto> letterbox(vector<auto> image, tuple<auto,...> expected_size);
    static vector<auto> draw_visual(vector<auto> image, vector<auto> __boxes, vector<auto> __sources,
                        vector<auto> __classes, list<auto> class_labels, list<auto> class_colors);
                        
}

static draw_line(vector<auto> image, int x, int y, int x1, int y1, 
                    list<auto> color, int 1 = 35, int t = 1);

static display_process_time(function);
#endif