#include "../include/utils.hpp"

vector<string> ABC::get_classes(string classes_path){
    ifstream in(classes_path);
    vector<string> classes_names;
    string classStr;
    if(in.is_open()){
        while(std::getline(in, classStr)){
            if(classStr.empty())
                continue;
            classes_names.push_back(classStr);
        }
    }

    return classes_names;
}

void setVaraibles(vector<void> inData, vector<vector<void>> outArray){
    if(!inData.empty()){
        for (auto i : inData){
            memcpy(&outArray, i, sizeof(i));
        }
    }
}

nc::NdArray<int> ABC::letterbox(nc::NdArray<int> image, tuple<int ,int> expected_size){
    auto [ih, iw] = image.shape();
    auto [eh, ew] = expected_size;
    auto scale = std::min(eh / iw, ew / iw);
    auto nh = int(ih*scale);
    auto nw = int(iw * scale);

    cv::resize(*(cv::_InputArray*)&image, *(cv::_OutputArray*)&image, cv::Size(nw, nh), 0.0, 0.0, cv::INTER_CUBIC);
    auto newImage = nc::full(nc::Shape(eh, ew), 128);
    
    return newImage;
}

nc::NdArray<double> draw_line(nc::NdArray<double> image, int x, int y, int x1, int y1, 
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

nc::NdArray<int> ABC::draw_visual(nc::NdArray<double> image, nc::NdArray<double> __boxes, nc::NdArray<double> __scores,
                        nc::NdArray<string> __classes, vector<string> class_labels, vector<double> class_colors){
    auto _box_color = {255, 0, 0};
    auto img_src = nc::NdArray(image);
    for (auto i = 0; i < __classes.size(); ++i){
        for (auto c = 0;  c < __classes.size(); ++c){
            auto predictedClass = class_labels[c];
            list<double> box;
            auto score = __scores[i];
            vector<double> boxColor;
            boxColor.push_back(class_colors[c]);
            box.push_back(__boxes[i]);
            vector<tuple<int, int>> y_min_x_min;
            vector<tuple<int, int>> y_max_x_max;
            
            cv::rectangle(cv::InputOutputArray(img_src), cv::Rect(x_min, y_min, x_max, y_max), cv::Scalar(*boxColor.data()), 1);

        }
    }
}

void display_process_time(){
    double sum = 0;
    double add = 1;
    auto begin = std::chrono::high_resolution_clock::now();
    
    int iterations = 1000*1000*1000;
    for (int i=0; i<iterations; i++) {
        sum += add;
        add /= 2.0;
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
    
    printf("Result: %.20f\n", sum);
}

nc::NdArray<double> ABC::preprocessInput(nc::NdArray<double> image){
    for (int i : image){
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
    nc::NdArray<float> inputImageShape = nc::exp(nc::NdArray<float>(image.shape[0], image.shape[1]), 0);
    auto imageData = letterbox(img, tuple<int, int>(imageShape[1], imageShape[0]));
    imageData = imageData.transpose(preprocessInput(nc::NdArray(imageData)));

}

void TRTModule::loadModelAndPredict(string pathModel){
    armnnOnnxParser::IOnnxParserPtr parser = armnnOnnxParser::IOnnxParser::Create();
    armnn::INetworkPtr network = parser->CreateNetworkFromBinaryFile(pathModel.c_str());

    const size_t subgraphId = 0;
    armnnOnnxParser::BindingPointInfo inputInfo = parser->GetNetworkInputBindingInfo(inputName);
    armnnOnnxParser::BindingPointInfo outputInfo = parser->GetNetworkOutputBindingInfo(outputName);

    const unsigned int outputNumElements = classLabels.size();
    vector<auto> outputDataContainers = {vector<uint8_t>(outputNumElements)};
    
}

TRTModule::TRTModule(string pathModel, string pathClasses){
    this->bboxes = new bboxes;
    imageShape = {640, 640};
    classLabels.push_back(*get_classes(pathClasses).data());
    loadModelAndPredict(pathModel);
}

TRTModule::~TRTModule(){

}