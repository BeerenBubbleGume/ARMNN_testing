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
    
    return newImage;
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

numcpp::ndarray ABC::draw_visual(numcpp::ndarray image, mvector<2, double> __boxes, numcpp::ndarray __scores,
                        numcpp::ndarray __classes, vector<string> class_labels, vector<string> class_colors){
    auto _box_color = {255, 0, 0};
    auto img_src = numcpp::array(image);
    for (auto i = 0; __classes.ptr() != nullptr; ++i){
        for (auto c = 0; __classes.ptr() != nullptr; ++c){
            auto predictedClass = class_labels[c];
            list<double> box;
            auto score = __scores[i];
            vector<string> boxColor;
            boxColor.push_back(class_colors[c]);
            box.push_back(__boxes);
            auto [y_min, x_min] = box.data();
            cv::rectangle(img_src, cv::Rect(x_min, y_min, x_max, y_max), cv::Scalar(_box_color.begin()), 1);

        }
    }
}

std::function<int(float)> display_process_time(function<int(float)> func){

    float decorated(int* args, float* argv[]){
        auto start = time::perf_counter();
    }
}

numcpp::ndarray ABC::preprocessInput(numcpp::ndarray image){
    for (auto i = 0; i < *image.get_shape(); ++i){
        image[i] = image[i] / 255.0;
    }
    return image;
}

map<string, vector<vector<void>>> TRTModule::trtInference(vector<vector<void>> intpuData, vector<vector<void>> imgz){
    
}
void TRTModule::startNN(string videoSrc, string outputPath, int fps){
    auto cap = cv::VideoCapture(videoSrc);
    vector<cv::UMat> frame;
    cap.read(frame);
    int width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    auto prevFrameTime = 0;
    auto fourcc = cv::VideoWriter::fourcc('M','J','P','G');
    auto out = cv::VideoWriter(outputPath, fourcc, fps, cv::Size(width, height));

    do{
        if(!cap.read(frame))
            break;
        auto output = extractImage(frame);
        auto newFrameTime = std::get_time();
        double FPS = 1 / (newFrameTime - prevFrameTime);
        prevFrameTime = newFrameTime;
        cv::putText(frame, std::to_string(FPS), cv::Point(5, 30), cv::FONT_HERSHEY_SIMPLEX,
                    0.5, cv::Scalar(0.0, 255.0, 255.0), 1);
        out.write(output);
    }while(cap.isOpened());
    out.release();
}

vector<vector<void>> TRTModule::extractImage(auto img){
    vector<cv::UMat> _img = img;
    
}

void TRTModule::loadModelAndPredict(string pathModel){
    armnnOnnxParser::IOnnxParserPtr parser = armnnOnnxParser::IOnnxParser::Create();
    armnn::INetworkPtr network = parser->CreateNetworkFromBinaryFile(pathModel.c_str());

    const size_t subgraphId = 0;
    armnnOnnxParser::BindingPointInfo inputInfo = parser->GetNetworkInputBindingInfo(subgraphId, inputName);
    armnnOnnxParser::BindingPointInfo outputInfo = parser->GetNetworkOutputBindingInfo(subgraphId, outputName);

    const unsigned int outputNumElements = modelOutputLabels.size();
    vector<TContainer> outputDataContainers = {vector<uint8_t>(outputNumElements)};

}

TRTModule::TRTModule(string pathModel, string pathClasses){
    bboxes = new bboxes;
    imageShape{640, 640};
    classLabels.push_back(get_classes(pathClasses));

}

TRTModule::~TRTModule(){

}