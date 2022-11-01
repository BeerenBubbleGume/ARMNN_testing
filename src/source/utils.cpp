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

nc::NdArray<float> ABC::letterbox(nc::NdArray<float> image, tuple<int ,int> expected_size){
    auto [ih, iw] = image.shape();
    auto [eh, ew] = expected_size;
    auto scale = std::min(eh / iw, ew / iw);
    auto nh = int(ih*scale);
    auto nw = int(iw * scale);

    cv::resize(*(cv::_InputArray*)&image, *(cv::_OutputArray*)&image, cv::Size(nw, nh), 0.0, 0.0, cv::INTER_CUBIC);
    nc::NdArray<float> newImage = nc::full(nc::Shape(eh, ew), (float)128.0);
    
    return newImage;
}

nc::NdArray<float> draw_line(nc::NdArray<float> image, int x, int y, int x1, int y1, 
                    vector<float> color, int l, int t){
    cv::line((cv::InputOutputArray)image, cv::Point(x, y), cv::Point(x + l, y), cv::Scalar_<float>(color[0], color[1], color[3]), t);
    cv::line((cv::InputOutputArray)image, cv::Point(x, y), cv::Point(x, y + l), cv::Scalar_<float>(color[0], color[1], color[3]), t);
    cv::line((cv::InputOutputArray)image, cv::Point(x1, y), cv::Point(x1 - l, y), cv::Scalar_<float>(color[0], color[1], color[3]), t);
    cv::line((cv::InputOutputArray)image, cv::Point(x1, y), cv::Point(x1, y + l), cv::Scalar_<float>(color[0], color[1], color[3]), t);
    cv::line((cv::InputOutputArray)image, cv::Point(x, y1), cv::Point(x + l, y1), cv::Scalar_<float>(color[0], color[1], color[3]), t);
    cv::line((cv::InputOutputArray)image, cv::Point(x, y1), cv::Point(x, y1 - l), cv::Scalar_<float>(color[0], color[1], color[3]), t);
    cv::line((cv::InputOutputArray)image, cv::Point(x1, y1), cv::Point(x1 - l, y1), cv::Scalar_<float>(color[0], color[1], color[3]), t);
    cv::line((cv::InputOutputArray)image, cv::Point(x1, y1), cv::Point(x1, y1 - l), cv::Scalar_<float>(color[0], color[1], color[3]), t);
    return image;
}

nc::NdArray<float> ABC::draw_visual(nc::NdArray<float> image, nc::NdArray<float> __boxes, nc::NdArray<float> __scores,
                        nc::NdArray<float> __classes, vector<string> class_labels, vector<float> class_colors){
    list<double> _box_color = {255., 0., 0.};
    auto img_src = nc::NdArray(image);
    for (auto i = 0; i < static_cast<int>(__classes.size()); ++i){
        for (auto c = 0;  c < static_cast<int>(__classes.size()); ++c){
            auto predictedClass = class_labels[c];
            vector<int> box;
            auto score = __scores[i];
            vector<float> boxColor;
            boxColor.push_back(class_colors[c]);
            box.push_back(__boxes[i]);
            int y_min, x_min, y_max, x_max;
            y_min = box[i]; x_min = box[i + 1]; y_max = box[i + 2]; x_max = box[i + 3];
            cv::rectangle(cv::InputOutputArray(img_src), cv::Rect(x_min, y_min, x_max, y_max), cv::Scalar(*boxColor.data()), 1);
            draw_line(img_src, x_min, y_min, x_max, y_max, boxColor);
            cv::putText(cv::InputOutputArray(img_src), predictedClass, cv::Point(x_min, y_min - 5), 
                        cv::FONT_HERSHEY_SIMPLEX, 0.35, cv::Scalar(0.0,255.0,255.0), 1);
        }
    }
    return img_src;
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

nc::NdArray<float> ABC::preprocessInput(nc::NdArray<float> image){
    for (int i : image){
        image[i] = image[i] / 255.0;
    }
    return image;
}

vector<nc::NdArray<float>> TRTModule::trtInference(nc::NdArray<float> inputData, nc::NdArray<float> imgz){
    nc::NdArray<float> ortInputs{inputData[inputData.none(), inputData.rSlice(), inputData.rSlice(), inputData.rSlice()]};
    armnn::Status ret = runtime->get()->EnqueueWorkload(*networkIdentifier,
      armnnUtils::MakeInputTensors(inputBindings, inputDataContainers),
      armnnUtils::MakeOutputTensors(outputBindings, outputDataContainers));
    box->preprocess(ortInputs, imageShape, imgz);
}
void TRTModule::startNN(string videoSrc, string outputPath, int fps){
    auto cap = cv::VideoCapture(videoSrc);
    nc::NdArray<float> frame;
    cap.read(cv::_OutputArray(frame.toStlVector()));
    auto width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    auto height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    std::chrono::duration<double> prevFrameTime = std::chrono::system_clock::now().time_since_epoch();
    auto fourcc = cv::VideoWriter::fourcc('M','J','P','G');
    auto out = cv::VideoWriter(outputPath, fourcc, fps, cv::Size(width, height));

    do{
        if(!cap.read(cv::_OutputArray(frame.toStlVector())))
            break;
        auto output = extractImage(frame);
        std::chrono::duration<double> newFrameTime = std::chrono::duration<double>(prevFrameTime);
        double FPS = 1 / (newFrameTime.count() - prevFrameTime.count());
        prevFrameTime = newFrameTime;
        cv::putText(cv::_InputOutputArray(frame.toStlVector()), std::to_string(FPS), cv::Point(5, 30), cv::FONT_HERSHEY_SIMPLEX,
                    0.5, cv::Scalar(0.0, 255.0, 255.0), 1);
        out.write(cv::InputArray(output));
    }while(cap.isOpened());
    out.release();
}

nc::NdArray<float> TRTModule::extractImage(nc::NdArray<float> img){
    nc::NdArray<float> inputImageShape = nc::NdArray<float>(static_cast<int>(img.shape().cols, img.shape().rows));
    nc::NdArray<float> imageData = letterbox(img, tuple<float, float>(imageShape[1], imageShape[0]));
    imageData = nc::transpose(preprocessInput(nc::NdArray(imageData)));
    vector<nc::NdArray<float>> __boxes__classes__scores(trtInference(imageData, inputImageShape));

    auto image = draw_visual(img, __boxes__classes__scores[0], 
                            __boxes__classes__scores[1], 
                            __boxes__classes__scores[2], 
                            classLabels, classColors);
    return image;
}

void TRTModule::loadModelAndPredict(string pathModel){
    armnnOnnxParser::IOnnxParserPtr __parser = armnnOnnxParser::IOnnxParser::Create();
    armnn::INetworkPtr __network = __parser->CreateNetworkFromBinaryFile(pathModel.c_str());
    
    parser = &__parser;
    network = &__network;

    const size_t subgraphId = 0;
    armnnOnnxParser::BindingPointInfo inputInfo = parser->get()->GetNetworkInputBindingInfo(inputName);
    armnnOnnxParser::BindingPointInfo outputInfo = parser->get()->GetNetworkOutputBindingInfo(outputName);

    const unsigned int outputNumElements = classLabels.size();
    outputDataContainers = {vector<uint8_t>(outputNumElements)};

    armnn::IRuntimePtr __runtime = armnn::IRuntime::Create(options);
    armnn::IOptimizedNetworkPtr __optNet = armnn::Optimize(*__network, {armnn::Compute::CpuAcc, armnn::Compute::CpuRef}, __runtime->GetDeviceSpec());
    
    runtime = &__runtime;
    optNet = &__optNet;

    armnn::NetworkId networkIdentifier;
    runtime->get()->LoadNetwork(networkIdentifier, std::move(*optNet));
    
    armnn::Status ret = runtime->get()->EnqueueWorkload(networkIdentifier,
      armnnUtils::MakeInputTensors(inputBindings, inputDataContainers),
      armnnUtils::MakeOutputTensors(outputBindings, outputDataContainers));
    
    //vector<uint8_t> output = std::get<vector<uint8_t>>(outputDataContainers[0]);
    // size_t labelInd = std::distance(output.begin(), std::max_element(output.begin(), output.end()));
    // std::cout << "Prediction: ";
    // std::cout << modelOutputLabels[labelInd] << std::endl;
}

TRTModule::TRTModule(string pathModel, string pathClasses){
    box = new bboxes;
    imageShape = {640.f, 640.f};
    classColors = {0.f, 0.f, 255.f};
    classLabels.push_back(*get_classes(pathClasses).data());
    inputName += "conv2d_input";
    outputName += "activation_5/Softmax";
    inputTensorBatchSize = 32;
    inputTensorDataLayout = armnn::DataLayout::NHWC; 
    NormalizationParameters optParam;
    inputDataContainers = {PrepareImageTensor<uint8_t>(inVideoName, static_cast<int>(imageShape[0]), static_cast<int>(imageShape[1]), optParam, inputTensorBatchSize, inputTensorDataLayout)};
    loadModelAndPredict(pathModel);
}

TRTModule::~TRTModule(){

}