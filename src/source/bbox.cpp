#include "../include/bbox.hpp"

nc::NdArray<float> bboxes::yolo_correct_boxes(nc::NdArray<float>  box_xy, nc::NdArray<float>  box_wh, 
                                vector<float> input_shape, nc::NdArray<float> image_shape){
    auto boxYX = box_xy(box_xy.rSlice(), nc::Slice(-1));
    auto boxHW = box_wh(box_wh.rSlice(), nc::Slice(-1));
    auto inputShape = nc::NdArray<float>(input_shape);
    auto imageShape = nc::NdArray<float>(image_shape);
    auto newShape = nc::round(imageShape * nc::min(inputShape / imageShape));
    nc::NdArray<float> offset;
    nc::NdArray<float> scale;

    for (int i = 0; i < inputShape.size(); ++i){
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
    auto _imageShape = {imageShape, image_shape};
    boxes *= nc::concatenate(_imageShape, nc::Axis(-1));
    return boxes;
}

bboxes::bboxes(){

}
bboxes::~bboxes(){

}