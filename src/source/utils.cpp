#include "../include/utils.hpp"

tuple<list<string>, int> ABC::get_classes(string classes_path){
    ifstream in(classes_path);
    list<string> classes_names;
    string classStr;
    if(in.is_open()){
        while(std::getline(in, classStr)){
            if(classStr.empty())
                continue;
            classes_names.push_back(classStr);
        }
    }

    tuple<list<string>, int> ret(classes_names, classes_names.size());
    return ret;
}

numcpp::ndarray ABC::letterbox(numcpp::ndarray image, tuple<int, int> expected_size){
    auto [ih, iw] = tuple<int, int>(*(image.get_shape()), *(image.get_shape()));
    auto [eh, ew] = expected_size;
    auto scale = std::min(eh / iw, ew / iw);
    auto nh = int(ih*scale);
    auto nw = int(iw * scale);

    cv::resize(*(cv::_InputArray*)&image, *(cv::_OutputArray*)&image, cv::Size(nw, nh), 0.0, 0.0, cv::INTER_CUBIC);
    auto newImage = numcpp::array(boost::python::api::object((eh, ew, 3)), 
                                    numcpp::dtype('uint8'));
    
}

numcpp::ndarray draw_line(numcpp::ndarray image, int x, int y, int x1, int y1, 
                    list<double> color, int l, int t){
    cv::line((cv::InputOutputArray)image, cv::Point(x, y), cv::Point(x + l, y), cv::Scalar_(color), t);
    cv::line((cv::InputOutputArray)image, cv::Point(x, y), cv::Point(x, y + l), cv::Scalar_(color), t);
    cv::line((cv::InputOutputArray)image, cv::Point(x1, y), cv::Point(x1 - l, y), cv::Scalar_(color), t);
    cv::line((cv::InputOutputArray)image, cv::Point(x1, y), cv::Point(x1, y + l), cv::Scalar_(color), t);
    cv::line((cv::InputOutputArray)image, cv::Point(x, y1), cv::Point(x + l, y1), cv::Scalar_(color), t);
    cv::line((cv::InputOutputArray)image, cv::Point(x, y1), cv::Point(x, y1 - l), cv::Scalar_(color), t);
    cv::line((cv::InputOutputArray)image, cv::Point(x1, y1), cv::Point(x1 - l, y1), cv::Scalar_(color), t);
    cv::line((cv::InputOutputArray)image, cv::Point(x1, y1), cv::Point(x1, y1 - l), cv::Scalar_(color), t);
    return image;
}

numcpp::ndarray ABC::draw_visual(numcpp::ndarray image, numcpp::ndarray __boxes, numcpp::ndarray __sources,
                        numcpp::ndarray __classes, list<string> class_labels, list<string> class_colors){

}

std::function<int(float)> display_process_time(function<int(float)> func){

    
}

numcpp::ndarray ABC::preprocessInput(numcpp::ndarray image){
    for (auto i = 0; i < *image.get_shape(); ++i){
        image[i] = image[i] / 255.0;
    }
    return image;
}