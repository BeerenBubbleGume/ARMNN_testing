#include "../include/bbox.hpp"

nc::NdArray<int> bboxes::yolo_correct_boxes(nc::NdArray<int>  box_xy, nc::NdArray<int>  box_wh, 
                                vector<int> input_shape, vector<int> image_shape){
    auto boxYX = box_xy(box_xy.rSlice(), nc::Slice(-1));
    auto boxHW = box_wh(box_wh.rSlice(), nc::Slice(-1));
    auto inputShape = nc::NdArray<int>(input_shape);
    auto imageShape = nc::NdArray<int>(image_shape);
    auto newShape = nc::round(imageShape * nc::min(inputShape / imageShape));
    nc::NdArray<int> offset;
    nc::NdArray<int> scale;

    for (auto i : inputShape){
        offset[i] = (inputShape[i] - newShape[i]) / 2. / inputShape[i];
        scale[i] = inputShape[i] / newShape[i];
    }

    boxYX = (box_xy - offset) * scale;
    boxHW *= scale;

    auto boxMinis = boxYX - (boxHW / 2);
    auto boxMaxes = boxYX + (boxHW / 2);
    auto listBoxPoints = {boxMinis(boxMinis.rSlice(), nc::Slice(0, 1)), 
                                    boxMinis(boxMinis.rSlice(), nc::Slice(1, 2)),
                                    boxMaxes(boxMaxes.rSlice(), nc::Slice(0, 1)),
                                    boxMaxes(boxMaxes.rSlice(), nc::Slice(1, 2))};
    auto boxes = nc::concatenate(listBoxPoints, -1);
    list<nc::NdArray<int>> _imageShape = {imageShape, image_shape};
    boxes *= nc::concatenate(_imageShape, -1);
    return boxes;
}

bboxes::bboxes(){

}
bboxes::~bboxes(){

}