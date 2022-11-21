#include "../include/bbox.hpp"

nc::NdArray<float> bboxes::yolo_correct_boxes(nc::NdArray<float>  box_xy, nc::NdArray<float>  box_wh, 
                                vector<float> input_shape, list<float> image_shape){
    auto boxYX = box_xy(box_xy.rSlice(), nc::Slice(-1));
    auto boxHW = box_wh(box_wh.rSlice(), nc::Slice(-1));
    auto inputShape = nc::NdArray<float>(input_shape);
    auto imageShape = nc::NdArray<float>(image_shape.begin(), image_shape.end());
    auto newShape = nc::round(imageShape * nc::min(inputShape / imageShape));
    nc::NdArray<float> offset;
    nc::NdArray<float> scale;

    for (int i = 0; i < static_cast<int>(inputShape.size()); ++i){
        offset[i] = (inputShape[i] - newShape[i]) / 2.f / inputShape[i];
        scale[i] = inputShape[i] / newShape[i];
    }

    boxYX = (box_xy - offset) * scale;
    boxHW *= scale;

    auto boxMinis = boxYX - (boxHW / 2.f);
    auto boxMaxes = boxYX + (boxHW / 2.f);
    auto listBoxPoints = {boxMinis(boxMinis.rSlice(), nc::Slice(0, 1)), 
                                    boxMinis(boxMinis.rSlice(), nc::Slice(1, 2)),
                                    boxMaxes(boxMaxes.rSlice(), nc::Slice(0, 1)),
                                    boxMaxes(boxMaxes.rSlice(), nc::Slice(1, 2))};
    auto boxes = nc::concatenate(listBoxPoints, nc::Axis(-1));
    auto _imageShape = {imageShape, nc::NdArray<float>(image_shape.begin(), image_shape.end())};
    boxes *= nc::concatenate(_imageShape, nc::Axis(-1));
    return boxes;
}

vector<nc::NdArray<float>> bboxes::preprocess(vector<Ort::Value> output, vector<float> image_data, list<float> image_shape){
    Ort::Value &pred = output.at(0);
    auto predDims = output.data()->GetTensorTypeAndShapeInfo().GetShape();
    auto numAncors = predDims.at(1);
    nc::NdArray<float> outputData;
    for (auto i = 0; i < numAncors; ++i){
        outputData.toStlVector().push_back(pred.At<float>({0, i, 0}));
        outputData.toStlVector().push_back(pred.At<float>({0, i, 1}));
        outputData.toStlVector().push_back(pred.At<float>({0, i, 2}));
        outputData.toStlVector().push_back(pred.At<float>({0, i, 3}));
    }
    auto boxXY = (outputData(outputData.rSlice(), {0, 2}) + outputData(outputData.rSlice(), {2, 4})) / 2.f;
    auto boxWH = outputData(outputData.rSlice(), {2, 4}) - outputData(nc::NdArray<float>(outputData).rSlice(), {0, 2});
    
    outputData(outputData.rSlice(), outputData.rSlice(4)) 
                    = yolo_correct_boxes(boxXY, boxWH, image_data, image_shape);
    vector<nc::NdArray<float>> ret = {outputData(outputData.rSlice(), outputData.rSlice(4)), 
                                        outputData(outputData.rSlice(), 
                                        outputData.rSlice(4)) * outputData(outputData.rSlice(),outputData.rSlice(5)),
                                        outputData(outputData.rSlice(), outputData.rSlice(6))};
    
    return ret;
}

bboxes::bboxes(){

}
bboxes::~bboxes(){

}