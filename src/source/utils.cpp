#include "../include/utils.hpp"

tuple<list<string>, int> ABC::get_classes(string classes_path){
    ifstream in(classes_path);
    list<string> classes_names;
    if(in.is_open())
        std::getline(in, classes_names);
    for (auto c : classes_names)
        classes_names.push_back(c);

    return (tuple<list<string>, int> ret(classes_names, classes_names.size()));
}

vector<auto> ABC::letterbox(vector<auto> image, tuple<auto,...> expected_size){
    int [ih, iw] = image.data();
}